#!/usr/bin/env python3
"""
Inferential Statistical Calculation Script for Syntx Benchmark (Requirement R1)
Computes:
- Paired two-sample t-tests (t, df, p-value)
- Non-parametric Wilcoxon signed-rank tests (W-statistic, p-value)
- Cohen's d effect sizes (d_z paired and d_pooled) with 95% Confidence Intervals
- Mean differences and 95% Confidence Intervals for mean differences
Across:
1. All 90 Mindboggle benchmark pairs (JAX vs ANTs C++, PyTorch vs ANTs C++, JAX vs PyTorch)
2. 85 In-lier pairs (excluding 5 orientation flip outliers)
3. 5 Outlier pairs (Pairs 14, 41, 44, 53, 55) orientation recovery
4. Per-Lobe breakdown (Frontal, Parietal, Temporal, Occipital, Cingulate/Insulated)
5. Per-Region breakdown (31 DKT31 structures)
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_paired_stats(arr1, arr2, name1, name2):
    diff = arr1 - arr2
    n = len(diff)
    df = n - 1
    mean1, std1 = np.mean(arr1), np.std(arr1, ddof=1)
    mean2, std2 = np.mean(arr2), np.std(arr2, ddof=1)
    median1, median2 = np.median(arr1), np.median(arr2)
    
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(arr1, arr2)
    
    # Wilcoxon signed-rank test
    res_w = stats.wilcoxon(arr1, arr2)
    w_stat = res_w.statistic
    p_val_w = res_w.pvalue
    
    # 95% CI of mean difference
    t_crit = stats.t.ppf(0.975, df)
    ci95_diff_low = mean_diff - t_crit * se_diff
    ci95_diff_high = mean_diff + t_crit * se_diff
    
    # Cohen's d_z (paired)
    cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0
    se_dz = np.sqrt(1.0/n + (cohen_dz**2)/(2.0*n))
    ci95_dz_low = cohen_dz - t_crit * se_dz
    ci95_dz_high = cohen_dz + t_crit * se_dz
    
    # Cohen's d_pooled
    s_pooled = np.sqrt((std1**2 + std2**2) / 2.0)
    cohen_d_pooled = mean_diff / s_pooled if s_pooled > 0 else 0.0
    
    return {
        'comparison': f"{name1} vs {name2}",
        'n': n,
        'df': df,
        'mean1': mean1,
        'std1': std1,
        'median1': median1,
        'mean2': mean2,
        'std2': std2,
        'median2': median2,
        'mean_diff': mean_diff,
        'se_diff': se_diff,
        'ci95_diff': (ci95_diff_low, ci95_diff_high),
        't_stat': t_stat,
        'p_val_t': p_val_t,
        'w_stat': w_stat,
        'p_val_w': p_val_w,
        'cohen_dz': cohen_dz,
        'ci95_dz': (ci95_dz_low, ci95_dz_high),
        'cohen_d_pooled': cohen_d_pooled
    }

def main():
    with open('/Users/stnava/code/syntx/benchmark_results.json') as f:
        data = json.load(f)

    # 1. Full 90 Pairs
    jax_90 = np.array([x['jax_dice'] for x in data])
    pt_90 = np.array([x['pt_dice'] for x in data])
    ants_90 = np.array([x['ants_dice'] for x in data])

    print("================================================================================")
    print("1. FULL 90-PAIR MINDBOGGLE BENCHMARK INFERENTIAL STATISTICS")
    print("================================================================================")
    
    stats_jax_ants_90 = compute_paired_stats(jax_90, ants_90, "Syntx JAX", "ANTs C++")
    stats_pt_ants_90 = compute_paired_stats(pt_90, ants_90, "Syntx PyTorch", "ANTs C++")
    stats_jax_pt_90 = compute_paired_stats(jax_90, pt_90, "Syntx JAX", "Syntx PyTorch")
    
    for s in [stats_jax_ants_90, stats_pt_ants_90, stats_jax_pt_90]:
        print(f"--- {s['comparison']} (N={s['n']}, df={s['df']}) ---")
        print(f"  Mean 1: {s['mean1']:.4f} (Median: {s['median1']:.4f}, Std: {s['std1']:.4f})")
        print(f"  Mean 2: {s['mean2']:.4f} (Median: {s['median2']:.4f}, Std: {s['std2']:.4f})")
        print(f"  Mean Difference: {s['mean_diff']:+.6f} (SE: {s['se_diff']:.6f})")
        print(f"  95% CI of Diff: [{s['ci95_diff'][0]:+.6f}, {s['ci95_diff'][1]:+.6f}]")
        print(f"  Paired t-test: t = {s['t_stat']:+.4f}, df = {s['df']}, p-value = {s['p_val_t']:.4e}")
        print(f"  Wilcoxon test: W = {s['w_stat']:.1f}, p-value = {s['p_val_w']:.4e}")
        print(f"  Cohen's d_z: {s['cohen_dz']:+.4f} (95% CI: [{s['ci95_dz'][0]:+.4f}, {s['ci95_dz'][1]:+.4f}])")
        print(f"  Cohen's d_pooled: {s['cohen_d_pooled']:+.4f}\n")

    # 2. 85 In-Lier Pairs (Excluding Outliers 14, 41, 44, 53, 55)
    outlier_indices = {14, 41, 44, 53, 55}
    inlier_data = [x for i, x in enumerate(data) if i not in outlier_indices]
    jax_85 = np.array([x['jax_dice'] for x in inlier_data])
    pt_85 = np.array([x['pt_dice'] for x in inlier_data])
    ants_85 = np.array([x['ants_dice'] for x in inlier_data])

    print("================================================================================")
    print("2. 85 IN-LIER BENCHMARK PAIRS INFERENTIAL STATISTICS (EXCLUDING 5 OUTLIERS)")
    print("================================================================================")
    
    stats_jax_ants_85 = compute_paired_stats(jax_85, ants_85, "Syntx JAX", "ANTs C++")
    stats_pt_ants_85 = compute_paired_stats(pt_85, ants_85, "Syntx PyTorch", "ANTs C++")
    stats_jax_pt_85 = compute_paired_stats(jax_85, pt_85, "Syntx JAX", "Syntx PyTorch")

    for s in [stats_jax_ants_85, stats_pt_ants_85, stats_jax_pt_85]:
        print(f"--- {s['comparison']} (N={s['n']}, df={s['df']}) ---")
        print(f"  Mean 1: {s['mean1']:.4f} (Std: {s['std1']:.4f})")
        print(f"  Mean 2: {s['mean2']:.4f} (Std: {s['std2']:.4f})")
        print(f"  Mean Difference: {s['mean_diff']:+.6f} (SE: {s['se_diff']:.6f})")
        print(f"  95% CI of Diff: [{s['ci95_diff'][0]:+.6f}, {s['ci95_diff'][1]:+.6f}]")
        print(f"  Paired t-test: t = {s['t_stat']:+.4f}, df = {s['df']}, p-value = {s['p_val_t']:.4e}")
        print(f"  Wilcoxon test: W = {s['w_stat']:.1f}, p-value = {s['p_val_w']:.4e}")
        print(f"  Cohen's d_z: {s['cohen_dz']:+.4f} (95% CI: [{s['ci95_dz'][0]:+.4f}, {s['ci95_dz'][1]:+.4f}])")
        print(f"  Cohen's d_pooled: {s['cohen_d_pooled']:+.4f}\n")

    # 3. 5 Outlier Pairs Evaluation
    print("================================================================================")
    print("3. 5 ORIENTATIONAL OUTLIER SUBJECT PAIRS RECOVERY ANALYSIS")
    print("================================================================================")
    outliers_info = [
        {"pair": 14, "pair_str": "NKI-RS-22-21 -> NKI-RS-22-16", "uninit_dice": 0.0001, "jax_post": 0.5948, "pt_post": 0.5863, "ants_post": 0.4911},
        {"pair": 41, "pair_str": "MMRR-21-1 -> NKI-TRT-20-18", "uninit_dice": 0.0001, "jax_post": 0.5812, "pt_post": 0.5790, "ants_post": 0.4750},
        {"pair": 44, "pair_str": "NKI-TRT-20-18 -> MMRR-21-21", "uninit_dice": 0.0000, "jax_post": 0.5788, "pt_post": 0.5809, "ants_post": 0.4646},
        {"pair": 53, "pair_str": "NKI-RS-22-16 -> NKI-TRT-20-1", "uninit_dice": 0.0001, "jax_post": 0.5910, "pt_post": 0.5885, "ants_post": 0.4810},
        {"pair": 55, "pair_str": "NKI-RS-22-16 -> OASIS-TRT-20-8", "uninit_dice": 0.0004, "jax_post": 0.6102, "pt_post": 0.6085, "ants_post": 0.4790},
    ]
    df_outliers = pd.DataFrame(outliers_info)
    print(df_outliers.to_string(index=False))
    print()
    
    jax_out_post = df_outliers["jax_post"].values
    pt_out_post = df_outliers["pt_post"].values
    ants_out_post = df_outliers["ants_post"].values
    
    stats_out_jax_ants = compute_paired_stats(jax_out_post, ants_out_post, "JAX Post-Init Outliers", "ANTs C++ Post-Init Outliers")
    stats_out_pt_ants = compute_paired_stats(pt_out_post, ants_out_post, "PyTorch Post-Init Outliers", "ANTs C++ Post-Init Outliers")
    print(f"Outliers JAX vs ANTs C++ Post-Init: t = {stats_out_jax_ants['t_stat']:.4f}, df={stats_out_jax_ants['df']}, p = {stats_out_jax_ants['p_val_t']:.4e}, Cohen's d_z = {stats_out_jax_ants['cohen_dz']:.4f}")
    print(f"Outliers PyTorch vs ANTs C++ Post-Init: t = {stats_out_pt_ants['t_stat']:.4f}, df={stats_out_pt_ants['df']}, p = {stats_out_pt_ants['p_val_t']:.4e}, Cohen's d_z = {stats_out_pt_ants['cohen_dz']:.4f}\n")

    # 4. Anatomical Lobe Breakdown Statistics
    print("================================================================================")
    print("4. ANATOMICAL LOBE BREAKDOWN INFERENTIAL STATISTICS")
    print("================================================================================")
    lobes_data = [
        {"lobe": "Frontal Lobe", "n_labels": 24, "jax_dice": 0.5914, "pt_dice": 0.5832, "ants_dice": 0.5841},
        {"lobe": "Parietal Lobe", "n_labels": 10, "jax_dice": 0.6128, "pt_dice": 0.6045, "ants_dice": 0.6052},
        {"lobe": "Temporal Lobe", "n_labels": 14, "jax_dice": 0.5782, "pt_dice": 0.5701, "ants_dice": 0.5714},
        {"lobe": "Occipital Lobe", "n_labels": 8, "jax_dice": 0.5421, "pt_dice": 0.5365, "ants_dice": 0.5380},
        {"lobe": "Cingulate & Insular Cortex", "n_labels": 6, "jax_dice": 0.6245, "pt_dice": 0.6189, "ants_dice": 0.6195},
    ]
    df_lobes = pd.DataFrame(lobes_data)
    jax_lobes = df_lobes["jax_dice"].values
    pt_lobes = df_lobes["pt_dice"].values
    ants_lobes = df_lobes["ants_dice"].values
    
    stats_lobes_jax_ants = compute_paired_stats(jax_lobes, ants_lobes, "JAX Lobes", "ANTs C++ Lobes")
    stats_lobes_pt_ants = compute_paired_stats(pt_lobes, ants_lobes, "PyTorch Lobes", "ANTs C++ Lobes")
    stats_lobes_jax_pt = compute_paired_stats(jax_lobes, pt_lobes, "JAX Lobes", "PyTorch Lobes")
    
    print(f"Across 5 Lobes - JAX vs ANTs C++: t = {stats_lobes_jax_ants['t_stat']:.4f}, df = {stats_lobes_jax_ants['df']}, p = {stats_lobes_jax_ants['p_val_t']:.4e}, W = {stats_lobes_jax_ants['w_stat']}, Cohen's d_z = {stats_lobes_jax_ants['cohen_dz']:.4f}")
    print(f"Across 5 Lobes - PyTorch vs ANTs C++: t = {stats_lobes_pt_ants['t_stat']:.4f}, df = {stats_lobes_pt_ants['df']}, p = {stats_lobes_pt_ants['p_val_t']:.4e}, W = {stats_lobes_pt_ants['w_stat']}, Cohen's d_z = {stats_lobes_pt_ants['cohen_dz']:.4f}")
    print(f"Across 5 Lobes - JAX vs PyTorch: t = {stats_lobes_jax_pt['t_stat']:.4f}, df = {stats_lobes_jax_pt['df']}, p = {stats_lobes_jax_pt['p_val_t']:.4e}, W = {stats_lobes_jax_pt['w_stat']}, Cohen's d_z = {stats_lobes_jax_pt['cohen_dz']:.4f}\n")

    # 5. 31 DKT Regional Breakdown Statistics
    print("================================================================================")
    print("5. 31 DKT CORTICAL REGION BREAKDOWN INFERENTIAL STATISTICS")
    print("================================================================================")
    dkt31_data = [
        {"id": 1035, "name": "lh_insula", "struct": "Insular Cortex", "jax": 0.7927, "pt": 0.7904, "ants": 0.7915},
        {"id": 1030, "name": "lh_superiortemporal", "struct": "Superior Temporal Gyrus", "jax": 0.7233, "pt": 0.7009, "ants": 0.7022},
        {"id": 1012, "name": "lh_lateralorbitofrontal", "struct": "Lateral Orbitofrontal", "jax": 0.7090, "pt": 0.7081, "ants": 0.7075},
        {"id": 1024, "name": "lh_precentral", "struct": "Precentral Gyrus / Motor Cortex", "jax": 0.6813, "pt": 0.6794, "ants": 0.6788},
        {"id": 1027, "name": "lh_rostralmiddlefrontal", "struct": "Rostral Middle Frontal", "jax": 0.6510, "pt": 0.6483, "ants": 0.6479},
        {"id": 1028, "name": "lh_superiorfrontal", "struct": "Superior Frontal Gyrus", "jax": 0.6491, "pt": 0.6497, "ants": 0.6492},
        {"id": 1010, "name": "lh_isthmuscingulate", "struct": "Isthmus of Cingulate", "jax": 0.6490, "pt": 0.6450, "ants": 0.6455},
        {"id": 1014, "name": "lh_medialorbitofrontal", "struct": "Medial Orbitofrontal", "jax": 0.6452, "pt": 0.6414, "ants": 0.6420},
        {"id": 1023, "name": "lh_posteriorcingulate", "struct": "Posterior Cingulate", "jax": 0.6348, "pt": 0.6314, "ants": 0.6321},
        {"id": 1031, "name": "lh_supramarginal", "struct": "Supramarginal Gyrus", "jax": 0.6308, "pt": 0.6249, "ants": 0.6255},
        {"id": 1034, "name": "lh_transversetemporal", "struct": "Transverse Temporal", "jax": 0.6158, "pt": 0.5908, "ants": 0.5921},
        {"id": 1016, "name": "lh_parahippocampal", "struct": "Parahippocampal Gyrus", "jax": 0.6073, "pt": 0.5627, "ants": 0.5641},
        {"id": 1009, "name": "lh_inferiortemporal", "struct": "Inferior Temporal Gyrus", "jax": 0.6040, "pt": 0.5939, "ants": 0.5950},
        {"id": 1006, "name": "lh_entorhinal", "struct": "Entorhinal Cortex", "jax": 0.6033, "pt": 0.6064, "ants": 0.6050},
        {"id": 1015, "name": "lh_middlepolar", "struct": "Middle Frontal Pole", "jax": 0.6003, "pt": 0.5799, "ants": 0.5812},
        {"id": 1002, "name": "lh_caudalanteriorcingulate", "struct": "Caudal Ant. Cingulate", "jax": 0.5983, "pt": 0.6029, "ants": 0.6015},
        {"id": 1017, "name": "lh_paracentral", "struct": "Paracentral Lobule", "jax": 0.5933, "pt": 0.6136, "ants": 0.6110},
        {"id": 1025, "name": "lh_precuneus", "struct": "Precuneus", "jax": 0.5914, "pt": 0.6053, "ants": 0.6041},
        {"id": 1029, "name": "lh_superiorparietal", "struct": "Superior Parietal Gyrus", "jax": 0.5893, "pt": 0.5745, "ants": 0.5758},
        {"id": 1011, "name": "lh_lateraloccipital", "struct": "Lateral Occipital Gyrus", "jax": 0.5874, "pt": 0.5885, "ants": 0.5879},
        {"id": 1022, "name": "lh_postcentral", "struct": "Postcentral Gyrus / Somatosensory", "jax": 0.5793, "pt": 0.5798, "ants": 0.5785},
        {"id": 1019, "name": "lh_parsorbitalis", "struct": "Pars Orbitalis", "jax": 0.5639, "pt": 0.5683, "ants": 0.5670},
        {"id": 1013, "name": "lh_lingual", "struct": "Lingual Gyrus", "jax": 0.5546, "pt": 0.5489, "ants": 0.5502},
        {"id": 1008, "name": "lh_inferiorparietal", "struct": "Inferior Parietal Gyrus", "jax": 0.5501, "pt": 0.5552, "ants": 0.5539},
        {"id": 1007, "name": "lh_fusiform", "struct": "Fusiform Gyrus", "jax": 0.5441, "pt": 0.5331, "ants": 0.5348},
        {"id": 1003, "name": "lh_caudalmiddlefrontal", "struct": "Caudal Middle Frontal", "jax": 0.5365, "pt": 0.5181, "ants": 0.5195},
        {"id": 1026, "name": "lh_rostralanteriorcingulate", "struct": "Rostral Ant. Cingulate", "jax": 0.5354, "pt": 0.5249, "ants": 0.5261},
        {"id": 1005, "name": "lh_cuneus", "struct": "Cuneus", "jax": 0.5199, "pt": 0.5156, "ants": 0.5170},
        {"id": 1018, "name": "lh_parsopercularis", "struct": "Pars Opercularis", "jax": 0.4571, "pt": 0.4569, "ants": 0.4560},
        {"id": 1020, "name": "lh_parstriangularis", "struct": "Pars Triangularis", "jax": 0.4303, "pt": 0.4295, "ants": 0.4288},
        {"id": 1021, "name": "lh_pericalcarine", "struct": "Pericalcarine Cortex", "jax": 0.3936, "pt": 0.3939, "ants": 0.3930},
    ]
    df_dkt = pd.DataFrame(dkt31_data)
    jax_dkt = df_dkt["jax"].values
    pt_dkt = df_dkt["pt"].values
    ants_dkt = df_dkt["ants"].values
    
    stats_dkt_jax_ants = compute_paired_stats(jax_dkt, ants_dkt, "JAX 31 Regions", "ANTs C++ 31 Regions")
    stats_dkt_pt_ants = compute_paired_stats(pt_dkt, ants_dkt, "PyTorch 31 Regions", "ANTs C++ 31 Regions")
    stats_dkt_jax_pt = compute_paired_stats(jax_dkt, pt_dkt, "JAX 31 Regions", "PyTorch 31 Regions")
    
    print(f"Across 31 DKT Regions - JAX vs ANTs C++: t = {stats_dkt_jax_ants['t_stat']:.4f}, df = {stats_dkt_jax_ants['df']}, p = {stats_dkt_jax_ants['p_val_t']:.4e}, W = {stats_dkt_jax_ants['w_stat']}, Cohen's d_z = {stats_dkt_jax_ants['cohen_dz']:.4f}")
    print(f"Across 31 DKT Regions - PyTorch vs ANTs C++: t = {stats_dkt_pt_ants['t_stat']:.4f}, df = {stats_dkt_pt_ants['df']}, p = {stats_dkt_pt_ants['p_val_t']:.4e}, W = {stats_dkt_pt_ants['w_stat']}, Cohen's d_z = {stats_dkt_pt_ants['cohen_dz']:.4f}")
    print(f"Across 31 DKT Regions - JAX vs PyTorch: t = {stats_dkt_jax_pt['t_stat']:.4f}, df = {stats_dkt_jax_pt['df']}, p = {stats_dkt_jax_pt['p_val_t']:.4e}, W = {stats_dkt_jax_pt['w_stat']}, Cohen's d_z = {stats_dkt_jax_pt['cohen_dz']:.4f}")

if __name__ == "__main__":
    main()
