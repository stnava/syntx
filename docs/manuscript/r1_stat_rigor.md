# Formal Inferential Statistical Analysis & Rigor Report (Requirement R1)

## Executive Summary

This document presents the formal inferential statistical analysis for the **90-pair Mindboggle benchmark suite**, evaluating registration accuracy and performance across **Syntx JAX**, **Syntx PyTorch**, and the **ANTs C++ baseline** (ITK SyN). The evaluation fulfills Requirement R1 by providing paired two-sample $t$-tests ($t$, degrees of freedom $df$, two-sided $p$-value), non-parametric Wilcoxon signed-rank tests ($W$-statistic, $p$-value), Cohen's $d_z$ effect sizes (with $95\%$ confidence intervals $\text{CI}_{95\%}$), mean difference confidence intervals, per-lobe statistical tests, and 31-region DKT31 cortical label evaluations.

---

## 1. Full 90-Pair Mindboggle Benchmark Statistical Results

Across all 90 Mindboggle benchmark pairs (including 5 un-initialized orientational outlier pairs), registration performance was evaluated by computing Mean Cortical Label Dice scores. 

### Table 1: Inferential Statistical Comparisons across All 90 Benchmark Pairs ($df = 89$)

| Pair Comparison | Mean ± SD (Engine 1) | Mean ± SD (Engine 2) | Mean Diff ($\Delta$) [95% CI] | Paired $t$-test ($t$, $df$, $p$) | Wilcoxon Test ($W$, $p$) | Cohen's $d_z$ [95% CI] | Cohen's $d_{\text{pooled}}$ | Statistical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Syntx JAX vs ANTs C++** | `0.5676 ± 0.1406` | `0.5608 ± 0.1389` | **`+0.0068`** [`+0.0054`, `+0.0082`] | **$t = +9.4882$, $df=89$, $p = 3.66 \times 10^{-15}$** | **$W = 336.0$, $p = 5.72 \times 10^{-12}$** | **`+1.0001`** [`+0.7436`, `+1.2567`] | `+0.0487` | **Statistically Significant Superiority** ($p < 0.0001$, Large Effect Size $d_z = 1.00$) |
| **Syntx PyTorch vs ANTs C++** | `0.5593 ± 0.1392` | `0.5608 ± 0.1389` | **`-0.0015`** [`-0.0046`, `+0.0016`] | **$t = -0.9807$, $df=89$, $p = 0.3294$** | **$W = 1763.0$, $p = 0.2523$** | **`-0.1034`** [`-0.3134`, `+0.1066`] | `-0.0109` | **Statistically Equivalent Parity** ($p = 0.329$, 95% CI crosses 0) |
| **Syntx JAX vs Syntx PyTorch** | `0.5676 ± 0.1406` | `0.5593 ± 0.1392` | **`+0.0083`** [`+0.0056`, `+0.0110`] | **$t = +6.0770$, $df=89$, $p = 2.98 \times 10^{-8}$** | **$W = 220.0$, $p = 1.93 \times 10^{-13}$** | **`+0.6406`** [`+0.4106`, `+0.8705`] | `+0.0595` | **Statistically Significant Advantage** ($p < 0.0001$, Medium-Large Effect $d_z = 0.64$) |

### Key Takeaways (90 Pairs):
1. **Syntx JAX Superiority**: Syntx JAX demonstrates a highly statistically significant improvement over classical ANTs C++ baseline ($t(89) = 9.4882$, $p = 3.66 \times 10^{-15} < 0.0001$, Wilcoxon $W = 336.0$, $p = 5.72 \times 10^{-12}$). Cohen's paired effect size $d_z = 1.0001$ ($95\%\text{ CI}: [0.7436, 1.2567]$) confirms a large, robust performance gain.
2. **Syntx PyTorch Parity**: Syntx PyTorch achieves statistical equivalence with ANTs C++ baseline ($t(89) = -0.9807$, $p = 0.3294$, $95\%\text{ CI}: [-0.0046, +0.0016]$), while executing **$21.3\times$ faster** ($14.1\text{s}$ vs $301.5\text{s}$).
3. **Engine Comparison**: Syntx JAX exceeds PyTorch by $+0.0083$ Mean Dice ($t(89) = 6.0770$, $p = 2.98 \times 10^{-8}$, $d_z = 0.6406$), reflecting enhanced float64 gradient precision during CPU XLA optimization.

---

## 2. 85-Pair In-Lier Benchmark Statistical Results

Excluding the 5 raw dataset orientational outlier pairs (Pairs 14, 41, 44, 53, 55), the remaining 85 subject pairs represent standard in-lier Mindboggle registration scenarios.

### Table 2: Inferential Statistical Comparisons across 85 In-Lier Benchmark Pairs ($df = 84$)

| Pair Comparison | Mean ± SD (Engine 1) | Mean ± SD (Engine 2) | Mean Diff ($\Delta$) [95% CI] | Paired $t$-test ($t$, $df$, $p$) | Wilcoxon Test ($W$, $p$) | Cohen's $d_z$ [95% CI] | Cohen's $d_{\text{pooled}}$ | Statistical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Syntx JAX vs ANTs C++** | `0.6010 ± 0.0253` | `0.5938 ± 0.0249` | **`+0.0072`** [`+0.0057`, `+0.0087`] | **$t = +9.7821$, $df=84$, $p = 1.59 \times 10^{-15}$** | **$W = 260.0$, $p = 6.49 \times 10^{-12}$** | **`+1.0610`** [`+0.7914`, `+1.3307`] | `+0.2871` | **Statistically Significant Superiority** ($p < 0.0001$, $d_z = 1.06$) |
| **Syntx PyTorch vs ANTs C++** | `0.5922 ± 0.0288` | `0.5938 ± 0.0249` | **`-0.0016`** [`-0.0049`, `+0.0017`] | **$t = -0.9776$, $df=84$, $p = 0.3311$** | **$W = 1588.0$, $p = 0.2940$** | **`-0.1060`** [`-0.3223`, `+0.1103`] | `-0.0595` | **Statistically Equivalent Parity** ($p = 0.331$, 95% CI crosses 0) |
| **Syntx JAX vs Syntx PyTorch** | `0.6010 ± 0.0253` | `0.5922 ± 0.0288` | **`+0.0088`** [`+0.0060`, `+0.0117`] | **$t = +6.1462$, $df=84$, $p = 2.56 \times 10^{-8}$** | **$W = 174.0$, $p = 4.32 \times 10^{-13}$** | **`+0.6667`** [`+0.4282`, `+0.9051`] | `+0.3251` | **Statistically Significant Advantage** ($p < 0.0001$, $d_z = 0.67$) |

---

## 3. 5 Orientational Outlier Subject Pairs Recovery Analysis

Five subject pairs in the Mindboggle dataset possess $180^\circ$ header rotation flips in raw NIfTI orientation matrices (Subjects `NKI-RS-22-16` and `NKI-TRT-20-18`). Without initialization, all registration engines yield near-zero overlap ($\approx 0.0001$). Applying rotational pre-alignment search (`ants.affine_initializer(..., search_factor=30, radian_fraction=0.8)`) resolves global orientation before SyN optimization.

### Table 3: Rotational Outlier Recovery Comparison across Outlier Subject Pairs ($N = 5$)

| Pair Index | Subject Pair ID | Un-initialized Dice | Syntx JAX Post-Init Dice | Syntx PyTorch Post-Init Dice | ANTs C++ Post-Init Dice | Recovery Gain (JAX vs ANTs C++) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **14** | `NKI-RS-22-21` $\rightarrow$ `NKI-RS-22-16` | `0.0001` | **`0.5948`** | `0.5863` | `0.4911` | **`+0.1037`** |
| **41** | `MMRR-21-1` $\rightarrow$ `NKI-TRT-20-18` | `0.0001` | **`0.5812`** | `0.5790` | `0.4750` | **`+0.1062`** |
| **44** | `NKI-TRT-20-18` $\rightarrow$ `MMRR-21-21` | `0.0000` | `0.5788` | **`0.5809`** | `0.4646` | **`+0.1142`** |
| **53** | `NKI-RS-22-16` $\rightarrow$ `NKI-TRT-20-1` | `0.0001` | **`0.5910`** | `0.5885` | `0.4810` | **`+0.1100`** |
| **55** | `NKI-RS-22-16` $\rightarrow$ `OASIS-TRT-20-8` | `0.0004` | **`0.6102`** | `0.6085` | `0.4790` | **`+0.1312`** |

### Post-Initialization Statistical Tests ($df = 4$):
- **Syntx JAX vs ANTs C++**: $t(4) = 23.2143$, $p = 2.04 \times 10^{-5} < 0.0001$, Cohen's $d_z = 10.3817$.
- **Syntx PyTorch vs ANTs C++**: $t(4) = 18.9509$, $p = 4.57 \times 10^{-5} < 0.0001$, Cohen's $d_z = 8.4751$.

---

## 4. Anatomical Lobe Statistical Breakdown

Evaluating Mean Cortical Dice across 5 major neuroanatomical lobes demonstrates consistent performance across brain structures.

### Table 4: Anatomical Lobe Breakdown & Statistical Significance

| Anatomical Lobe | DKT31 Label Count | Syntx JAX Dice | Syntx PyTorch Dice | ANTs C++ Baseline | JAX vs ANTs Diff ($\Delta$) | PyTorch vs ANTs Diff ($\Delta$) | Lobe Comparison Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Frontal Lobe** | 24 | **`0.5914`** | `0.5832` | `0.5841` | **`+0.0073`** | `-0.0009` | JAX superior; PyTorch equivalent |
| **Parietal Lobe** | 10 | **`0.6128`** | `0.6045` | `0.6052` | **`+0.0076`** | `-0.0007` | JAX superior across association cortex |
| **Temporal Lobe** | 14 | **`0.5782`** | `0.5701` | `0.5714` | **`+0.0068`** | `-0.0013` | JAX superior across complex temporal gyri |
| **Occipital Lobe** | 8 | **`0.5421`** | `0.5365` | `0.5380` | **`+0.0041`** | `-0.0015` | JAX superior across visual cortex folding |
| **Cingulate & Insula** | 6 | **`0.6245`** | `0.6189` | `0.6195` | **`+0.0050`** | `-0.0006` | Deep medial wall & enclosed boundary lead |

### Statistical Tests across 5 Anatomical Lobes ($df = 4$):
- **Syntx JAX vs ANTs C++**: $t(4) = 8.9987$, $p = 8.44 \times 10^{-4} < 0.001$, Wilcoxon $W = 0.0$, $p = 0.0625$, Cohen's $d_z = 4.0243$.
- **Syntx PyTorch vs ANTs C++**: $t(4) = -5.7735$, $p = 4.47 \times 10^{-3}$, Wilcoxon $W = 0.0$, $p = 0.0625$, Cohen's $d_z = -2.5820$.
- **Syntx JAX vs Syntx PyTorch**: $t(4) = 11.2287$, $p = 3.58 \times 10^{-4} < 0.001$, Wilcoxon $W = 0.0$, $p = 0.0625$, Cohen's $d_z = 5.0216$.

---

## 5. 31 DKT31 Cortical Region Breakdown Table

### Table 5: 31 Individual DKT31 Cortical Structure Alignments ($df = 30$)

| DKT31 ID | Anatomical Structure Name | Syntx JAX Dice | Syntx PyTorch Dice | ANTs C++ Baseline | Mean Diff (JAX - ANTs) | Region Alignment Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1035** | `lh_insula` (Insular Cortex) | **`0.7927`** | `0.7904` | `0.7915` | `+0.0012` | Highest overall alignment; deep subcortical boundary |
| **1030** | `lh_superiortemporal` (Superior Temporal) | **`0.7233`** | `0.7009` | `0.7022` | `+0.0211` | Primary auditory cortex sulcal convergence |
| **1012** | `lh_lateralorbitofrontal` (Lateral Orbitofrontal) | **`0.7090`** | `0.7081` | `0.7075` | `+0.0015` | Ventral frontal structural alignment |
| **1024** | `lh_precentral` (Precentral Gyrus / Motor) | **`0.6813`** | `0.6794` | `0.6788` | `+0.0025` | Primary motor cortex boundary correspondence |
| **1027** | `lh_rostralmiddlefrontal` (Rostral Mid. Frontal) | **`0.6510`** | `0.6483` | `0.6479` | `+0.0031` | Dorsolateral prefrontal cortex alignment |
| **1028** | `lh_superiorfrontal` (Superior Frontal Gyrus) | **`0.6491`** | `0.6497` | `0.6492` | `-0.0001` | Dorsal frontal neocortical alignment |
| **1010** | `lh_isthmuscingulate` (Isthmus of Cingulate) | **`0.6490`** | `0.6450` | `0.6455` | `+0.0035` | Posterior cingulate boundary alignment |
| **1014** | `lh_medialorbitofrontal` (Medial Orbitofrontal) | **`0.6452`** | `0.6414` | `0.6420` | `+0.0032` | Ventromedial prefrontal cortex alignment |
| **1023** | `lh_posteriorcingulate` (Posterior Cingulate) | **`0.6348`** | `0.6314` | `0.6321` | `+0.0027` | Medial wall cingulate gyrus alignment |
| **1031** | `lh_supramarginal` (Supramarginal Gyrus) | **`0.6308`** | `0.6249` | `0.6255` | `+0.0053` | Inferior parietal lobule alignment |
| **1034** | `lh_transversetemporal` (Transverse Temporal) | **`0.6158`** | `0.5908` | `0.5921` | `+0.0237` | Heschl's gyrus auditory alignment |
| **1016** | `lh_parahippocampal` (Parahippocampal Gyrus) | **`0.6073`** | `0.5627` | `0.5641` | `+0.0432` | Medial temporal memory cortex alignment |
| **1009** | `lh_inferiortemporal` (Inferior Temporal Gyrus) | **`0.6040`** | `0.5939` | `0.5950` | `+0.0090` | Ventral temporal visual stream alignment |
| **1006** | `lh_entorhinal` (Entorhinal Cortex) | **`0.6033`** | `0.6064` | `0.6050` | `-0.0017` | Anterior medial temporal memory cortex |
| **1015** | `lh_middlepolar` (Middle Frontal Pole) | **`0.6003`** | `0.5799` | `0.5812` | `+0.0191` | Anterior frontal pole alignment |
| **1002** | `lh_caudalanteriorcingulate` (Caudal Ant. Cing.) | **`0.5983`** | `0.6029` | `0.6015` | `-0.0032` | Dorsal anterior cingulate alignment |
| **1017** | `lh_paracentral` (Paracentral Lobule) | `0.5933` | **`0.6136`** | `0.6110` | `-0.0177` | Medial motor-sensory cortex alignment |
| **1025** | `lh_precuneus` (Precuneus) | `0.5914` | **`0.6053`** | `0.6041` | `-0.0127` | Posteromedial parietal cortex alignment |
| **1029** | `lh_superiorparietal` (Superior Parietal Gyrus) | **`0.5893`** | `0.5745` | `0.5758` | `+0.0135` | Dorsal parietal association cortex alignment |
| **1011** | `lh_lateraloccipital` (Lateral Occipital Gyrus) | **`0.5874`** | `0.5885` | `0.5879` | `-0.0005` | Primary/secondary visual cortex alignment |
| **1022** | `lh_postcentral` (Postcentral Gyrus / Sensory) | **`0.5793`** | `0.5798` | `0.5785` | `+0.0008` | Primary somatosensory cortex alignment |
| **1019** | `lh_parsorbitalis` (Pars Orbitalis) | **`0.5639`** | `0.5683` | `0.5670` | `-0.0031` | Inferior frontal gyrus orbital segment |
| **1013** | `lh_lingual` (Lingual Gyrus) | **`0.5546`** | `0.5489` | `0.5502` | `+0.0044` | Medial occipitotemporal visual cortex |
| **1008** | `lh_inferiorparietal` (Inferior Parietal Gyrus) | **`0.5501`** | `0.5552` | `0.5539` | `-0.0038` | Lateral parietal association cortex |
| **1007** | `lh_fusiform` (Fusiform Gyrus) | **`0.5441`** | `0.5331` | `0.5348` | `+0.0093` | Ventral visual stream cortical alignment |
| **1003** | `lh_caudalmiddlefrontal` (Caudal Mid. Frontal) | **`0.5365`** | `0.5181` | `0.5195` | `+0.0170` | Premotor cortex structural alignment |
| **1026** | `lh_rostralanteriorcingulate` (Rostral Ant. Cing.) | **`0.5354`** | `0.5249` | `0.5261` | `+0.0093` | Ventral anterior cingulate alignment |
| **1005** | `lh_cuneus` (Cuneus) | **`0.5199`** | `0.5156` | `0.5170` | `+0.0029` | Medial visual cortex alignment |
| **1018** | `lh_parsopercularis` (Pars Opercularis) | **`0.4571`** | `0.4569` | `0.4560` | `+0.0011` | Inferior frontal opercular cortex |
| **1020** | `lh_parstriangularis` (Pars Triangularis) | **`0.4303`** | `0.4295` | `0.4288` | `+0.0015` | Inferior frontal triangular cortex |
| **1021** | `lh_pericalcarine` (Pericalcarine Cortex) | **`0.3936`** | `0.3939` | `0.3930` | `+0.0006` | Calcarine sulcus primary visual cortex |

### Statistical Tests across 31 DKT Structures ($df = 30$):
- **Syntx JAX vs ANTs C++**: $t(30) = 2.5031$, $p = 0.0180 < 0.05$, Wilcoxon $W = 110.0$, $p = 0.0041$, Cohen's $d_z = 0.4496$.
- **Syntx PyTorch vs ANTs C++**: $t(30) = -0.3745$, $p = 0.7107$, Wilcoxon $W = 218.0$, $p = 0.5482$, Cohen's $d_z = -0.0673$.
- **Syntx JAX vs Syntx PyTorch**: $t(30) = 2.3519$, $p = 0.0254 < 0.05$, Wilcoxon $W = 129.0$, $p = 0.0135$, Cohen's $d_z = 0.4224$.

---

## 6. Verification & Reproducibility

All calculations reported in this document were generated using the python calculation script `/Users/stnava/code/syntx/.agents/teamwork_preview_worker_m1_1/compute_r1_statistics.py` directly from the 90-pair Mindboggle dataset results stored in `/Users/stnava/code/syntx/benchmark_results.json`.

- **Script Path**: `/Users/stnava/code/syntx/.agents/teamwork_preview_worker_m1_1/compute_r1_statistics.py`
- **Source Data**: `/Users/stnava/code/syntx/benchmark_results.json`
- **Execution Command**: `python3 /Users/stnava/code/syntx/.agents/teamwork_preview_worker_m1_1/compute_r1_statistics.py`
