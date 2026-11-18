# Medical Segmentation Decathlon (MSD) Provenance & Evaluation Report

**Date**: 2026-11-17 16:01:40  
**Device**: `mps`  
**Cohort Scale**: `5 pair(s) per task` (`10/10` tasks recorded)  
**Deformable Schedule**: `[100, 100, 20]`  
**Smoke Mode**: `False`  
**Standard**: GEMINI.md Section 6 (Whole-Organ vs Focal Lesion Separation & Hold-Out Landmark TRE)

## Stratum A: Anatomical Organ Alignment (6 Tasks)

Evaluates whole-organ parenchymal overlap (Sørensen-Dice) with combined parenchymal and lesion coverage.

| Task Name | Anatomy | Modality | Init Dice | ANTs Affine | syntx Native Affine | Winner | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task02_Heart** | Heart | MRI | 0.3056 | 0.5622 | 0.5857 | **Default** | 0.7688 | **0.8617** | **+0.0929** | **+0.5561** | 0.160 mm | 0.0000% | -0.0104 |
| **Task03_Liver** | Liver | CT | 0.0000 | 0.6473 | 0.6319 | **Default** | 0.7760 | **0.8694** | **+0.0934** | **+0.8694** | 0.213 mm | 0.0000% | -0.0177 |
| **Task04_Hippocampus** | Hippocampus | MRI | 0.4421 | 0.7218 | 0.7352 | **Default** | 0.7908 | **0.8047** | **+0.0140** | **+0.3626** | 0.693 mm | 0.0000% | 0.1347 |
| **Task05_Prostate** | Prostate | MRI | 0.2850 | 0.3097 | 0.5755 | **Default** | 0.3229 | **0.6761** | **+0.3531** | **+0.3910** | 0.262 mm | 0.0000% | 0.0153 |
| **Task07_Pancreas** | Pancreas | CT | 0.0808 | 0.2027 | 0.2529 | **Default** | 0.2929 | **0.4284** | **+0.1355** | **+0.3476** | 0.269 mm | 0.0001% | -0.0552 |
| **Task09_Spleen** | Spleen | CT | 0.0433 | 0.2617 | 0.2878 | **Default** | 0.4369 | **0.8013** | **+0.3644** | **+0.7580** | 0.244 mm | 0.0001% | -0.0632 |

## Stratum B: Surrogate Parenchymal Alignment for Focal Pathology (4 Tasks)

Evaluates surrogate parenchymal overlap (brain mask, lung cavity, abdominal soft tissue) and structural correlation.

| Task Name | Anatomy | Modality | Surrogate Definition | Init Dice | ANTs Affine | syntx Native Affine | ANTs SyN | syntx SyN Beltrami | Δ vs ANTs SyN | Net Gain (syntx) | Hold-Out TRE | Folding % | Min Jac (Liouville) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task01_BrainTumour** | Brain | MRI | surrogate: whole-brain mask, T1w structural channel | 0.9197 | 0.9427 | 0.9437 | 0.9778 | **0.9802** | **+0.0024** | **+0.0605** | 0.084 mm | 0.0000% | 0.0036 |
| **Task06_Lung** | Lung | CT | surrogate: bilateral lung parenchyma | 0.4127 | 0.6290 | 0.6247 | 0.7595 | **0.8131** | **+0.0537** | **+0.4004** | 0.192 mm | 0.0000% | 0.0025 |
| **Task08_HepaticVessel** | Hepatic Vessel | CT | surrogate: abdominal soft tissue | 0.6674 | 0.7180 | 0.7424 | 0.7811 | **0.8458** | **+0.0647** | **+0.1784** | 0.434 mm | 0.0000% | 0.0092 |
| **Task10_Colon** | Colon | CT | surrogate: abdominal soft tissue | 0.3524 | 0.6933 | 0.6710 | 0.7483 | **0.7612** | **+0.0129** | **+0.4088** | 0.337 mm | 0.0001% | -0.0342 |

## Stratum Summaries & Power Analysis

### Stratum A Summary (Anatomical Organ Tasks):
- **Mean Initial Dice**: `0.1928`
- **Mean Affine Dice**: `syntx` **`0.5115`** vs `ANTs` `0.4509` (`+0.0606`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.7403`** vs `ANTs SyN` `0.5647` (**`+0.1755`**, net gain **`+0.5475`**)
- **Mean Hold-Out Target Registration Error**: **`0.307 mm`**

### Stratum B Summary (Focal Lesion Tasks with Surrogates):
- **Mean Initial Surrogate Dice**: `0.5881`
- **Mean Affine Dice**: `syntx` **`0.7454`** vs `ANTs` `0.7458` (`-0.0003`)
- **Mean Deformable Dice**: `syntx SyN Beltrami` **`0.8501`** vs `ANTs SyN` `0.8167` (**`+0.0334`**, net gain **`+0.2620`**)
- **Mean Hold-Out Target Registration Error**: **`0.262 mm`**

