#!/usr/bin/env python3
"""
scripts/train_diagnostic_classifier.py — Train 3D ResNet-10 Multi-Task Diagnostic Classifier
=========================================================================================

Trains the Tier 2 deep diagnostic classifier on Medical Segmentation Decathlon (MSD) tasks:
- Predicts Anatomy: BRAIN, THORAX, ABDOMEN, PELVIS, HEART
- Predicts Modality: CT, MRI_T1, MRI_T2, MRI_FLAIR, MRI_ADC
"""

import os
import argparse
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import ants

from syntx.classifier import (
    DiagnosticClassifier3D,
    ANATOMY_TO_IDX,
    MODALITY_TO_IDX,
    preprocess_volume_for_classifier,
)
from syntx.data.msd import MSD_TASKS, MSDDataset


class MSDClassifierDataset(Dataset):
    """
    Multi-task classification dataset parsing MSD tasks into (volume, anatomy_idx, modality_idx).
    """

    def __init__(self, data_dir: str, target_shape=(64, 64, 64)):
        self.samples = []
        self.target_shape = target_shape

        for task_id, info in MSD_TASKS.items():
            task_dir = os.path.join(data_dir, info.name)
            if not os.path.exists(task_dir):
                task_dir = os.path.join(data_dir, task_id)
            meta_path = os.path.join(task_dir, "dataset.json")
            if not os.path.exists(meta_path):
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            anatomy = info.target_anatomy.upper()
            modality = info.primary_modality.upper()

            # Normalization / mapping
            if anatomy not in ANATOMY_TO_IDX:
                if "LUNG" in anatomy:
                    anatomy = "THORAX"
                elif "LIVER" in anatomy or "PANCREAS" in anatomy or "SPLEEN" in anatomy or "COLON" in anatomy:
                    anatomy = "ABDOMEN"
                else:
                    anatomy = "BRAIN"

            if modality not in MODALITY_TO_IDX:
                if "T1" in modality:
                    modality = "MRI_T1"
                elif "T2" in modality:
                    modality = "MRI_T2"
                elif "CT" in modality:
                    modality = "CT"
                else:
                    modality = "MRI_T1"

            anat_idx = ANATOMY_TO_IDX[anatomy]
            mod_idx = MODALITY_TO_IDX[modality]

            for case in meta.get("training", []):
                img_p = os.path.normpath(os.path.join(task_dir, case["image"]))
                if os.path.exists(img_p):
                    self.samples.append((img_p, anat_idx, mod_idx))

        print(f"MSDClassifierDataset: Loaded {len(self.samples)} cases across tasks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, anat_idx, mod_idx = self.samples[idx]
        img = ants.image_read(img_p)
        tensor = preprocess_volume_for_classifier(img, target_shape=self.target_shape).squeeze(0)  # (1, D, H, W)
        return tensor, torch.tensor(anat_idx, dtype=torch.long), torch.tensor(mod_idx, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train 3D ResNet-10 Diagnostic Classifier on MSD")
    parser.add_argument("--data-dir", type=str, default="/Users/stnava/data/decathlon",
                        help="Root directory where MSD tasks are stored")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="src/syntx/models/diagnostic_resnet10_3d.pth",
                        help="Path to save trained weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = MSDClassifierDataset(args.data_dir)
    if len(dataset) == 0:
        print("No cases found in data directory. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DiagnosticClassifier3D().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_anat = nn.CrossEntropyLoss()
    criterion_mod = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        correct_anat = 0
        correct_mod = 0
        total_items = 0

        for x, y_anat, y_mod in loader:
            x = x.to(device)
            y_anat = y_anat.to(device)
            y_mod = y_mod.to(device)

            optimizer.zero_grad()
            logits_anat, logits_mod = model(x)

            loss_anat = criterion_anat(logits_anat, y_anat)
            loss_mod = criterion_mod(logits_mod, y_mod)
            loss = loss_anat + loss_mod

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            correct_anat += (logits_anat.argmax(dim=1) == y_anat).sum().item()
            correct_mod += (logits_mod.argmax(dim=1) == y_mod).sum().item()
            total_items += len(x)

        ep_loss = total_loss / max(total_items, 1)
        ep_acc_anat = correct_anat / max(total_items, 1) * 100.0
        ep_acc_mod = correct_mod / max(total_items, 1) * 100.0
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} [{time.time()-t0:.1f}s] Loss: {ep_loss:.4f} | Anat Acc: {ep_acc_anat:.1f}% | Mod Acc: {ep_acc_mod:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nModel saved successfully to {args.output}")


if __name__ == "__main__":
    main()
