#!/usr/bin/env python3
"""
Automated Dynamic Manuscript Builder for Syntx Research
======================================================
1. Dynamically decodes all benchmark metrics, hypothesis tests, effect sizes,
   folding distributions, inverse errors, and runtimes directly from canonical CSVs.
2. Injects computed values into docs/manuscript/manuscript_template.md.
3. Renders docs/manuscript/manuscript_report.md with zero hardcoded specific values.
4. Compiles standalone HTML (with MathJax & Citeproc) and XeLaTeX PDF.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UNIFIED_CSV = os.path.join(ROOT_DIR, 'results', 'cohort_90pair_all_4methods_unified_summary.csv')
SYNGS_CSV = os.path.join(ROOT_DIR, 'results', 'cohort_90pair_syngs_sobolev_summary.csv')
TEMPLATE_MD = os.path.join(ROOT_DIR, 'docs', 'manuscript', 'manuscript_template.md')
REPORT_MD = os.path.join(ROOT_DIR, 'docs', 'manuscript', 'manuscript_report.md')
REPORT_HTML = os.path.join(ROOT_DIR, 'docs', 'manuscript', 'manuscript_report.html')
REPORT_PDF = os.path.join(ROOT_DIR, 'docs', 'manuscript', 'manuscript_report.pdf')
BIB_FILE = os.path.join(ROOT_DIR, 'docs', 'manuscript', 'references.bib')

def compute_all_metrics():
    df = pd.read_csv(UNIFIED_CSV)
    df_syngs = pd.read_csv(SYNGS_CSV)

    n_total = len(df)
    intra = df[df['type'] == 'intra']
    inter = df[df['type'] == 'inter']
    n_intra = len(intra)
    n_inter = len(inter)

    # Baseline affine
    aff_mean = df['dice_affine'].mean()
    aff_std = df['dice_affine'].std()
    aff_min = df['dice_affine'].min()
    aff_max = df['dice_affine'].max()
    aff_median = df['dice_affine'].median()

    # Methods
    d_ants = df['dice_ants'].values
    d_syn = df['dice_syn'].values
    d_syngs = df['dice_syngs'].values
    d_tvf = df['dice_tvf'].values

    # Folding
    fold_ants = df['fold_brain_ants'].mean()
    fold_syn = df['fold_brain_syn'].mean()
    fold_syngs = df['fold_brain_syngs'].mean()
    fold_tvf = df['fold_brain_tvf'].mean()

    # Time
    time_ants = df['time_ants'].mean()
    time_syn = df['time_syn'].mean()
    time_syngs = df['time_syngs'].mean()
    time_tvf = df['time_tvf'].mean()

    # Statistical tests & win rates
    def get_comparison_stats(arr, ref=d_ants):
        diff = arr - ref
        t_res = stats.ttest_rel(arr, ref)
        w_res = stats.wilcoxon(diff)
        cohen_d = np.mean(diff) / np.std(diff)
        wins = int(np.sum(diff > 1e-4))
        ties = int(np.sum(np.abs(diff) <= 1e-4))
        losses = int(np.sum(diff < -1e-4))
        win_rate = (wins / len(arr)) * 100.0
        gain_abs = np.mean(diff)
        gain_pct = gain_abs * 100.0
        return {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'gain_abs': gain_abs,
            'gain_pct': gain_pct,
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'win_rate': win_rate,
            't_stat': t_res.statistic,
            't_pval': t_res.pvalue,
            'w_stat': w_res.statistic,
            'w_pval': w_res.pvalue,
            'cohen_d': cohen_d
        }

    st_ants = {'mean': np.mean(d_ants), 'std': np.std(d_ants)}
    st_syn = get_comparison_stats(d_syn)
    st_syngs = get_comparison_stats(d_syngs)
    st_tvf = get_comparison_stats(d_tvf)

    # Subgroups
    sub_intra = {
        'ants': f"{intra['dice_ants'].mean():.4f} ± {intra['dice_ants'].std():.4f}",
        'syn': f"{intra['dice_syn'].mean():.4f} ± {intra['dice_syn'].std():.4f} (+{(intra['dice_syn'].mean()-intra['dice_ants'].mean())*100:.2f}%)",
        'syngs': f"{intra['dice_syngs'].mean():.4f} ± {intra['dice_syngs'].std():.4f} (+{(intra['dice_syngs'].mean()-intra['dice_ants'].mean())*100:.2f}%)",
        'tvf': f"{intra['dice_tvf'].mean():.4f} ± {intra['dice_tvf'].std():.4f}",
        'tvf_gain': f"+{(intra['dice_tvf'].mean()-intra['dice_ants'].mean())*100:.2f}%"
    }

    sub_inter = {
        'ants': f"{inter['dice_ants'].mean():.4f} ± {inter['dice_ants'].std():.4f}",
        'syn': f"{inter['dice_syn'].mean():.4f} ± {inter['dice_syn'].std():.4f} (+{(inter['dice_syn'].mean()-inter['dice_ants'].mean())*100:.2f}%)",
        'syngs': f"{inter['dice_syngs'].mean():.4f} ± {inter['dice_syngs'].std():.4f} (+{(inter['dice_syngs'].mean()-inter['dice_ants'].mean())*100:.2f}%)",
        'tvf': f"{inter['dice_tvf'].mean():.4f} ± {inter['dice_tvf'].std():.4f}",
        'tvf_gain': f"+{(inter['dice_tvf'].mean()-inter['dice_ants'].mean())*100:.2f}%"
    }

    sub_total = {
        'ants': f"{st_ants['mean']:.4f} ± {st_ants['std']:.4f}",
        'syn': f"{st_syn['mean']:.4f} ± {st_syn['std']:.4f} (+{st_syn['gain_pct']:.2f}%)",
        'syngs': f"{st_syngs['mean']:.4f} ± {st_syngs['std']:.4f} (+{st_syngs['gain_pct']:.2f}%)",
        'tvf': f"{st_tvf['mean']:.4f} ± {st_tvf['std']:.4f}",
        'tvf_gain': f"+{st_tvf['gain_pct']:.2f}%"
    }

    # Format p-values nicely
    def fmt_p(p):
        if p < 1e-4:
            return f"{p:.2e}".replace('e-0', 'e-').replace('e-', ' \\times 10^{-') + '}'
        return f"{p:.4f}"

    # Dictionary of decoded variables
    ctx = {
        'n_total': str(n_total),
        'n_intra': str(n_intra),
        'n_inter': str(n_inter),
        'aff_mean_std': f"{aff_mean:.4f} ± {aff_std:.4f}",
        'aff_min': f"{aff_min:.4f}",
        'aff_max': f"{aff_max:.4f}",
        'aff_median': f"{aff_median:.4f}",
        
        # ANTs
        'dice_ants_mean_std': f"{st_ants['mean']:.4f} ± {st_ants['std']:.4f}",
        'time_ants': f"{time_ants:.1f}",
        'fold_ants': f"{fold_ants:.4f}",
        
        # SyN
        'dice_syn_mean_std': f"{st_syn['mean']:.4f} ± {st_syn['std']:.4f}",
        'gain_syn_abs': f"{st_syn['gain_abs']:.4f}",
        'gain_syn_pct': f"{st_syn['gain_pct']:.2f}",
        'win_syn_record': f"{st_syn['wins']} / {n_total} ({st_syn['win_rate']:.1f}%)",
        't_syn_stat': f"{st_syn['t_stat']:.4f}",
        't_syn_pval': fmt_p(st_syn['t_pval']),
        'w_syn_stat': f"{st_syn['w_stat']:.1f}",
        'w_syn_pval': fmt_p(st_syn['w_pval']),
        'cohen_syn_d': f"{st_syn['cohen_d']:.4f}",
        'time_syn': f"{time_syn:.1f}",
        'speedup_syn': f"{time_ants/time_syn:.2f}",
        'fold_syn': f"{fold_syn:.4f}",
        
        # SyNGS
        'dice_syngs_mean_std': f"{st_syngs['mean']:.4f} ± {st_syngs['std']:.4f}",
        'gain_syngs_abs': f"{st_syngs['gain_abs']:.4f}",
        'gain_syngs_pct': f"{st_syngs['gain_pct']:.2f}",
        'win_syngs_record': f"{st_syngs['wins']} / {n_total} ({st_syngs['win_rate']:.1f}%)",
        't_syngs_stat': f"{st_syngs['t_stat']:.4f}",
        't_syngs_pval': fmt_p(st_syngs['t_pval']),
        'w_syngs_stat': f"{st_syngs['w_stat']:.1f}",
        'w_syngs_pval': fmt_p(st_syngs['w_pval']),
        'cohen_syngs_d': f"{st_syngs['cohen_d']:.4f}",
        'time_syngs': f"{time_syngs:.1f}",
        'speedup_syngs': f"{time_ants/time_syngs:.2f}",
        'fold_syngs': f"{fold_syngs:.4f}",
        
        # TVF
        'dice_tvf_mean_std': f"{st_tvf['mean']:.4f} ± {st_tvf['std']:.4f}",
        'gain_tvf_abs': f"{st_tvf['gain_abs']:.4f}",
        'gain_tvf_pct': f"{st_tvf['gain_pct']:.2f}",
        'win_tvf_record': f"{st_tvf['wins']} / {n_total} ({st_tvf['win_rate']:.1f}%)",
        't_tvf_stat': f"{st_tvf['t_stat']:.4f}",
        't_tvf_pval': fmt_p(st_tvf['t_pval']),
        'w_tvf_stat': f"{st_tvf['w_stat']:.1f}",
        'w_tvf_pval': fmt_p(st_tvf['w_pval']),
        'cohen_tvf_d': f"{st_tvf['cohen_d']:.4f}",
        'time_tvf': f"{time_tvf:.1f}",
        'speedup_tvf': f"{time_ants/time_tvf:.2f}",
        'fold_tvf': f"{fold_tvf:.4f}",
        
        # Subgroups
        'sub_intra_ants': sub_intra['ants'],
        'sub_intra_syn': sub_intra['syn'],
        'sub_intra_syngs': sub_intra['syngs'],
        'sub_intra_tvf': sub_intra['tvf'],
        'sub_intra_tvf_gain': sub_intra['tvf_gain'],
        
        'sub_inter_ants': sub_inter['ants'],
        'sub_inter_syn': sub_inter['syn'],
        'sub_inter_syngs': sub_inter['syngs'],
        'sub_inter_tvf': sub_inter['tvf'],
        'sub_inter_tvf_gain': sub_inter['tvf_gain'],
        
        'sub_total_ants': sub_total['ants'],
        'sub_total_syn': sub_total['syn'],
        'sub_total_syngs': sub_total['syngs'],
        'sub_total_tvf': sub_total['tvf'],
        'sub_total_tvf_gain': sub_total['tvf_gain'],
    }

    return ctx

def render_manuscript(ctx):
    print("Rendering manuscript from template...")
    with open(TEMPLATE_MD, 'r') as f:
        template = f.read()

    rendered = template
    for k, v in ctx.items():
        placeholder = '{{' + k + '}}'
        rendered = rendered.replace(placeholder, v)

    with open(REPORT_MD, 'w') as f:
        f.write(rendered)
    print(f"Saved rendered manuscript: {REPORT_MD}")

def compile_documents():
    print("Compiling HTML manuscript via Pandoc...")
    cmd_html = [
        'pandoc', REPORT_MD,
        '--citeproc',
        f'--bibliography={BIB_FILE}',
        '--resource-path=.:docs/manuscript',
        '--mathjax',
        '--standalone',
        '--toc',
        '--number-sections',
        '-o', REPORT_HTML
    ]
    subprocess.run(cmd_html, check=True)
    print(f"Saved HTML: {REPORT_HTML}")

    print("Compiling PDF manuscript via Pandoc + XeLaTeX...")
    cmd_pdf = [
        'pandoc', REPORT_MD,
        '--citeproc',
        f'--bibliography={BIB_FILE}',
        '--resource-path=.:docs/manuscript',
        '--pdf-engine=xelatex',
        '--number-sections',
        '-o', REPORT_PDF
    ]
    subprocess.run(cmd_pdf, check=True)
    print(f"Saved PDF: {REPORT_PDF}")

if __name__ == '__main__':
    ctx = compute_all_metrics()
    print("Computed metrics summary:")
    for k, v in list(ctx.items())[:15]:
        print(f"  {k}: {v}")
    render_manuscript(ctx)
    compile_documents()
    print("\nManuscript pipeline completed successfully!")
