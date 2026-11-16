# Medical Segmentation Decathlon (MSD) Provenance & Evaluation Report

**Date**: 2026-11-16 04:08:31  
**Device**: `mps`  
**Cohort Scale**: `5 pair(s) per task` (`10/10` tasks recorded)  
**Deformable Schedule**: `[100, 100, 20]`  
**Smoke Mode**: `False`  
**Standard**: GEMINI.md Section 6 (Whole-Organ vs Focal Lesion Separation & Hold-Out Landmark TRE)

## Stratum A: Anatomical Organ Alignment (6 Tasks)

Evaluates whole-organ parenchymal overlap (Sørensen-Dice) with combined parenchymal and lesion coverage.

| Task Name | Anatomy | Modality | Init Dice | ANTs Affine | syntx Native Affine | Winner | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task02_Heart** | Heart | MRI | 0.3056 | 0.5592 | 0.5607 | **Default** | 0.7691 | **0.8604** | **+0.0913** | **+0.5549** | 0.196 mm | 0.0076% | -0.0199 |
| **Task03_Liver** | Liver | CT | 0.0000 | 0.6631 | 0.6313 | **Default** | 0.7905 | **0.8688** | **+0.0783** | **+0.8688** | 0.264 mm | 0.0009% | -0.0266 |
| **Task04_Hippocampus** | Hippocampus | MRI | 0.4421 | 0.7107 | 0.7352 | **Default** | 0.7876 | **0.7990** | **+0.0114** | **+0.3569** | 0.689 mm | 0.0000% | 0.1473 |
| **Task05_Prostate** | Prostate | MRI | 0.2850 | 0.3742 | 0.5573 | **Default** | 0.3850 | **0.6741** | **+0.2892** | **+0.3891** | 0.306 mm | 0.0000% | 0.0145 |
| **Task07_Pancreas** | Pancreas | CT | 0.0808 | 0.2024 | 0.2664 | **Default** | 0.2953 | **0.4266** | **+0.1313** | **+0.3458** | 0.278 mm | 0.0012% | -0.0000 |
| **Task09_Spleen** | Spleen | CT | 0.0433 | 0.2703 | 0.2827 | **Default** | 0.4448 | **0.7834** | **+0.3386** | **+0.7401** | 0.364 mm | 0.0048% | -8.1346 |

## Stratum B: Surrogate Parenchymal Alignment for Focal Pathology (4 Tasks)

Evaluates surrogate parenchymal overlap (brain mask, lung cavity, abdominal soft tissue) and structural correlation.

| Task Name | Anatomy | Modality | Surrogate Definition | Init Dice | ANTs Affine | syntx Native Affine | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task01_BrainTumour** | Brain | MRI | surrogate: whole-brain mask, T1w structural channel | 0.9197 | 0.9420 | 0.9438 | 0.9780 | **0.9804** | **+0.0024** | **+0.0607** | 0.122 mm | 0.0000% | 0.0051 |
| **Task06_Lung** | Lung | CT | surrogate: bilateral lung parenchyma | 0.4127 | 0.6321 | 0.6247 | 0.7631 | **0.8132** | **+0.0500** | **+0.4005** | 0.304 mm | 0.0000% | 0.0025 |
| **Task08_HepaticVessel** | Hepatic Vessel | CT | surrogate: abdominal soft tissue | 0.6674 | 0.7193 | 0.7424 | 0.7821 | **0.8458** | **+0.0637** | **+0.1784** | 0.390 mm | 0.0000% | 0.0093 |
| **Task10_Colon** | Colon | CT | surrogate: abdominal soft tissue | 0.3524 | 0.7034 | 0.6711 | 0.7605 | **0.7615** | **+0.0011** | **+0.4091** | 0.374 mm | 0.0000% | -0.0156 |

## Stratum Summaries & Power Analysis

### Stratum A Summary (Anatomical Organ Tasks):
- **Mean Initial Dice**: `0.1928`
- **Mean Affine Dice**: `syntx` **`0.5056`** vs `ANTs` `0.4633` (`+0.0423`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.7354`** vs `ANTs SyN` `0.5787` (**`+0.1567`**, net gain **`+0.5426`**)
- **Mean Hold-Out Target Registration Error**: **`0.349 mm`**

### Stratum B Summary (Focal Lesion Tasks with Surrogates):
- **Mean Initial Surrogate Dice**: `0.5881`
- **Mean Affine Dice**: `syntx` **`0.7455`** vs `ANTs` `0.7492` (`-0.0037`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.8502`** vs `ANTs SyN` `0.8209` (**`+0.0293`**, net gain **`+0.2622`**)
- **Mean Hold-Out Target Registration Error**: **`0.298 mm`**

