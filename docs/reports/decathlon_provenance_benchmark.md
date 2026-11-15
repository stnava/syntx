# Medical Segmentation Decathlon (MSD) Provenance & Evaluation Report

**Date**: 2026-11-15 16:32:56  
**Device**: `mps`  
**Cohort Scale**: `5 pair(s) per task` (`10/10` tasks recorded)  
**Deformable Schedule**: `[100, 100, 20]`  
**Smoke Mode**: `False`  
**Standard**: GEMINI.md Section 6 (Whole-Organ vs Focal Lesion Separation & Hold-Out Landmark TRE)

## Stratum A: Anatomical Organ Alignment (6 Tasks)

Evaluates whole-organ parenchymal overlap (Sørensen-Dice) with combined parenchymal and lesion coverage.

| Task Name | Anatomy | Modality | Init Dice | ANTs Affine | syntx Native Affine | Winner | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task02_Heart** | Heart | MRI | 0.3056 | 0.5579 | 0.5757 | **Default** | 0.7712 | **0.7942** | **+0.0230** | **+0.4887** | 0.196 mm | 0.0075% | -0.0228 |
| **Task03_Liver** | Liver | CT | 0.0000 | 0.6611 | 0.6476 | **Default** | 0.7929 | **0.7840** | **-0.0089** | **+0.7840** | 0.264 mm | 0.0000% | 0.0065 |
| **Task04_Hippocampus** | Hippocampus | MRI | 0.4421 | 0.7220 | 0.7170 | **Default** | 0.7777 | **0.8033** | **+0.0256** | **+0.3612** | 1.058 mm | 0.0000% | 0.1941 |
| **Task05_Prostate** | Prostate | MRI | 0.2850 | 0.3092 | 0.4593 | **Default** | 0.3225 | **0.5395** | **+0.2170** | **+0.2545** | 0.301 mm | 0.0023% | -0.0000 |
| **Task07_Pancreas** | Pancreas | CT | 0.0808 | 0.2027 | 0.2725 | **Default** | 0.2934 | **0.3495** | **+0.0561** | **+0.2687** | 0.278 mm | 0.0000% | 0.0097 |
| **Task09_Spleen** | Spleen | CT | 0.0433 | 0.2688 | 0.2447 | **Default** | 0.4843 | **0.6040** | **+0.1197** | **+0.5607** | 0.364 mm | 0.0401% | -0.0000 |

## Stratum B: Surrogate Parenchymal Alignment for Focal Pathology (4 Tasks)

Evaluates surrogate parenchymal overlap (brain mask, lung cavity, abdominal soft tissue) and structural correlation.

| Task Name | Anatomy | Modality | Surrogate Definition | Init Dice | ANTs Affine | syntx Native Affine | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task01_BrainTumour** | Brain | MRI | surrogate: whole-brain mask, T1w structural channel | 0.9080 | 0.9296 | 0.9337 | 0.9654 | **0.9709** | **+0.0054** | **+0.0628** | 0.122 mm | 0.0000% | 0.0929 |
| **Task06_Lung** | Lung | CT | surrogate: bilateral lung parenchyma | 0.4127 | 0.6241 | 0.6147 | 0.7554 | **0.8019** | **+0.0465** | **+0.3892** | 0.304 mm | 0.0045% | -0.0000 |
| **Task08_HepaticVessel** | Hepatic Vessel | CT | surrogate: abdominal soft tissue | 0.6674 | 0.7337 | 0.7380 | 0.7947 | **0.7942** | **-0.0005** | **+0.1268** | 0.390 mm | 0.0000% | 0.0138 |
| **Task10_Colon** | Colon | CT | surrogate: abdominal soft tissue | 0.3524 | 0.6869 | 0.6945 | 0.7428 | **0.7371** | **-0.0057** | **+0.3847** | 0.374 mm | 0.0000% | 0.0053 |

## Stratum Summaries & Power Analysis

### Stratum A Summary (Anatomical Organ Tasks):
- **Mean Initial Dice**: `0.1928`
- **Mean Affine Dice**: `syntx` **`0.4861`** vs `ANTs` `0.4536` (`+0.0325`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.6457`** vs `ANTs SyN` `0.5737` (**`+0.0721`**, net gain **`+0.4529`**)
- **Mean Hold-Out Target Registration Error**: **`0.410 mm`**

### Stratum B Summary (Focal Lesion Tasks with Surrogates):
- **Mean Initial Surrogate Dice**: `0.5851`
- **Mean Affine Dice**: `syntx` **`0.7452`** vs `ANTs` `0.7436` (`+0.0016`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.8260`** vs `ANTs SyN` `0.8146` (**`+0.0114`**, net gain **`+0.2409`**)
- **Mean Hold-Out Target Registration Error**: **`0.298 mm`**

