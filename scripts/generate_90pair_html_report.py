#!/usr/bin/env python
"""
Generate comprehensive side-by-side HTML report with interactive X-Y scatter plots:
1. Syntx Dice vs. ANTs Dice (with parity line y=x)
2. Syntx Compute Time vs. ANTs Compute Time (with 1x, 2x, 3x speedup reference lines)
Outputs to: results/90pair_report.html
"""

import os
import sys
import json
import glob
import time
import numpy as np

def generate_html_report(results_dir="results/pairs_90/syn_mps", ants_dir="results", out_html="results/90pair_report.html"):
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    
    # Load all completed Syntx pair results
    pair_files = sorted(glob.glob(os.path.join(results_dir, "pair_*_syn.json")))
    if not pair_files:
        pair_files = sorted(glob.glob("results/pair_*_syn.json"))
        
    syntx_records = {}
    for pf in pair_files:
        try:
            with open(pf, "r") as f:
                d = json.load(f)
                if d.get("status") == "SUCCESS":
                    syntx_records[d["pair_idx"]] = d
        except Exception:
            pass

    # Load matching ANTs C++ baseline results
    ants_files = sorted(glob.glob(os.path.join(ants_dir, "pair_*_ants_syn.json")))
    ants_records = {}
    for af in ants_files:
        try:
            with open(af, "r") as f:
                d = json.load(f)
                if d.get("status") == "SUCCESS":
                    ants_records[d["pair_idx"]] = d
        except Exception:
            pass

    completed_indices = sorted(list(syntx_records.keys()))
    n_completed = len(completed_indices)
    
    # Matched comparison statistics
    matched_pairs = []
    for idx in completed_indices:
        s_rec = syntx_records[idx]
        a_rec = ants_records.get(idx, {})
        matched_pairs.append((idx, s_rec, a_rec))

    s_dices = [s["dice_sym"] for _, s, _ in matched_pairs]
    a_dices = [a.get("dice_sym", float('nan')) for _, _, a in matched_pairs if a.get("dice_sym") is not None]
    
    s_folds = [s.get("folding_pct", 0.0) for _, s, _ in matched_pairs]
    a_folds = [a.get("folding_pct", 0.0) for _, _, a in matched_pairs if a.get("folding_pct") is not None]
    
    s_times = [s.get("runtime_seconds", 0.0) for _, s, _ in matched_pairs]
    a_times = [a.get("runtime_seconds", 0.0) for _, _, a in matched_pairs if a.get("runtime_seconds") is not None]

    mean_s_dice = float(np.mean(s_dices)) if s_dices else 0.0
    mean_a_dice = float(np.nanmean(a_dices)) if a_dices else 0.0
    dice_diff = mean_s_dice - mean_a_dice
    
    wins = sum(1 for _, s, a in matched_pairs if a.get("dice_sym") is not None and s["dice_sym"] >= a["dice_sym"])
    total_compared = sum(1 for _, _, a in matched_pairs if a.get("dice_sym") is not None)
    win_rate = (wins / total_compared * 100.0) if total_compared > 0 else 0.0

    mean_s_fold = float(np.mean(s_folds)) if s_folds else 0.0
    mean_a_fold = float(np.nanmean(a_folds)) if a_folds else 0.0
    
    mean_s_time = float(np.mean(s_times)) if s_times else 0.0
    mean_a_time = float(np.nanmean(a_times)) if a_times else 0.0
    speedup = (mean_a_time / mean_s_time) if mean_s_time > 0 else 1.0
    
    total_time_h = float(np.sum(s_times)) / 3600.0 if s_times else 0.0

    # Data arrays for Plotly
    plot_pair_ids = [f"Pair {idx:02d} ({s.get('pair_type', 'intra').upper()})" for idx, s, _ in matched_pairs]
    plot_syn_dice = [round(s["dice_sym"], 4) for _, s, _ in matched_pairs]
    plot_ants_dice = [round(a["dice_sym"], 4) if a.get("dice_sym") is not None else None for _, _, a in matched_pairs]
    plot_syn_time = [round(s.get("runtime_seconds", 0), 1) for _, s, _ in matched_pairs]
    plot_ants_time = [round(a.get("runtime_seconds", 0), 1) if a.get("runtime_seconds") is not None else None for _, _, a in matched_pairs]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syntx vs. ANTs C++ — 90-Pair Mindboggle Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-dim: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: var(--font-family);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #ffffff;
            font-size: 26px;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            font-size: 13px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent);
            border: 1px solid rgba(88, 166, 255, 0.3);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
        }}
        .stat-sub {{
            font-size: 12px;
            color: var(--text-dim);
            margin-top: 4px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        h2 {{
            color: #ffffff;
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 16px;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{
            .plots-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .plot-box {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            height: 440px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: #21262d;
            color: #ffffff;
            font-weight: 600;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .pill {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .pill-intra {{
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
        }}
        .pill-inter {{
            background: rgba(210, 153, 34, 0.15);
            color: var(--accent-orange);
        }}
        .gain-pos {{
            color: var(--accent-green);
            font-weight: 600;
        }}
        .gain-neg {{
            color: var(--accent-red);
            font-weight: 600;
        }}
        .config-box {{
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
            font-family: monospace;
            font-size: 12px;
            color: #79c0ff;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Syntx vs. ANTs C++ &mdash; 90-Pair Mindboggle Population Benchmark <span class="badge">Autograd Gaussian Peak</span></h1>
            <div style="color: var(--text-dim); font-size: 13px;">
                Syntx: <code>syntx.syn (Eulerian + ITK Sampled Gaussian Kernel + Autograd) on GPU</code> &bull; Baseline: <code>ANTs C++ SyN on CPU</code> &bull; Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Syntx vs. ANTs Mean Dice</div>
                <div class="stat-value" style="color: var(--accent-green);">{mean_s_dice:.4f} <span style="font-size: 16px; color: var(--text-dim);">vs {mean_a_dice:.4f}</span></div>
                <div class="stat-sub">Advantage: <strong class="gain-pos">{dice_diff*100:+.2f}%</strong> ({wins}/{total_compared} Wins, {win_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Grid Folding Regularity</div>
                <div class="stat-value" style="color: var(--accent-green);">{mean_s_fold:.4f}%</div>
                <div class="stat-sub">ANTs C++ Baseline: {mean_a_fold:.4f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Runtime &amp; Speedup</div>
                <div class="stat-value" style="color: var(--accent-purple);">{mean_s_time:.1f}s <span style="font-size: 16px; color: var(--text-dim);">vs {mean_a_time:.1f}s</span></div>
                <div class="stat-sub"><strong class="gain-pos">{speedup:.2f}&times; Faster</strong> on Apple Silicon GPU</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Progress Throughput</div>
                <div class="stat-value" style="color: var(--accent);">{n_completed} / 90</div>
                <div class="stat-sub">Completed pairs ({total_time_h:.2f}h compute)</div>
            </div>
        </div>

        <div class="plots-grid">
            <div class="plot-box" id="diceScatterPlot"></div>
            <div class="plot-box" id="timeScatterPlot"></div>
        </div>

        <div class="card">
            <h2>Algorithm Provenance Configuration</h2>
            <div class="config-box">
Syntx Engine: formulation='eulerian' &bull; kernel_type='gaussian' (sampled ITK Gaussian, radius=floor(3σ+0.5))<br>
use_analytical_gradients=false (sliding box Autograd + flip physical scale) &bull; flow_sigma=3.0 (σ ≈ 1.732 voxels)<br>
total_sigma=0.0 &bull; grad_step=0.25 (multi-res sqrt(shrink_ratio)) &bull; inverse_method='anderson'<br>
similarity_metric='cc2' (LNCC window=5x5x5, Var_safe=1e-6) &bull; reg_iterations=[100, 100, 20]
            </div>
        </div>

        <div class="card">
            <h2>Per-Pair Side-by-Side Comparison ({n_completed} Completed)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pair</th>
                        <th>Type</th>
                        <th>Fixed Target ID</th>
                        <th>Moving Source ID</th>
                        <th>Syntx Dice</th>
                        <th>ANTs Dice</th>
                        <th>&Delta; Dice</th>
                        <th>Syntx Fold%</th>
                        <th>ANTs Fold%</th>
                        <th>Syntx Time</th>
                        <th>ANTs Time</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
"""

    for idx, s_rec, a_rec in matched_pairs:
        p_type = s_rec.get("pair_type", "intra")
        pill_cls = "pill-intra" if p_type == "intra" else "pill-inter"
        f_id = s_rec.get("fixed_id", f"pair_{idx}_fix")
        m_id = s_rec.get("moving_id", f"pair_{idx}_mov")
        
        s_dice = s_rec.get("dice_sym", 0.0)
        a_dice = a_rec.get("dice_sym", float('nan'))
        
        if not np.isnan(a_dice):
            diff = s_dice - a_dice
            diff_str = f"{diff*100:+.2f}%"
            diff_cls = "gain-pos" if diff >= 0 else "gain-neg"
            a_dice_str = f"{a_dice:.4f}"
        else:
            diff_str = "&mdash;"
            diff_cls = ""
            a_dice_str = "&mdash;"

        s_fold = s_rec.get("folding_pct", 0.0)
        a_fold = a_rec.get("folding_pct", float('nan'))
        a_fold_str = f"{a_fold:.4f}%" if not np.isnan(a_fold) else "&mdash;"
        
        s_t = s_rec.get("runtime_seconds", 0.0)
        a_t = a_rec.get("runtime_seconds", float('nan'))
        a_t_str = f"{a_t:.1f}s" if not np.isnan(a_t) else "&mdash;"
        
        sp_str = f"{a_t/s_t:.2f}&times;" if (not np.isnan(a_t) and s_t > 0) else "&mdash;"

        html += f"""                    <tr>
                        <td><strong>#{idx:02d}</strong></td>
                        <td><span class="pill {pill_cls}">{p_type.upper()}</span></td>
                        <td><code>{f_id}</code></td>
                        <td><code>{m_id}</code></td>
                        <td><strong style="color: var(--accent);">{s_dice:.4f}</strong></td>
                        <td>{a_dice_str}</td>
                        <td><span class="{diff_cls}">{diff_str}</span></td>
                        <td>{s_fold:.4f}%</td>
                        <td>{a_fold_str}</td>
                        <td>{s_t:.1f}s</td>
                        <td>{a_t_str}</td>
                        <td><strong class="gain-pos">{sp_str}</strong></td>
                    </tr>
"""

    html += f"""                </tbody>
            </table>
        </div>
    </div>

    <script>
        const pairLabels = {json.dumps(plot_pair_ids)};
        const synDice = {json.dumps(plot_syn_dice)};
        const antsDice = {json.dumps(plot_ants_dice)};
        const synTime = {json.dumps(plot_syn_time)};
        const antsTime = {json.dumps(plot_ants_time)};

        // 1. X-Y Scatter: Syntx Dice vs ANTs Dice
        const pairedAntsDice = [];
        const pairedSynDice = [];
        const pairedDiceLabels = [];
        for (let i = 0; i < pairLabels.length; i++) {{
            if (antsDice[i] !== null && synDice[i] !== null) {{
                pairedAntsDice.push(antsDice[i]);
                pairedSynDice.push(synDice[i]);
                const diff = (synDice[i] - antsDice[i]) * 100;
                pairedDiceLabels.push(pairLabels[i] + '<br>Syntx: ' + synDice[i] + '<br>ANTs: ' + antsDice[i] + '<br>&Delta;: ' + (diff >= 0 ? '+' : '') + diff.toFixed(2) + '%');
            }}
        }}

        const minDice = Math.min(...pairedAntsDice, ...pairedSynDice, 0.55);
        const maxDice = Math.max(...pairedAntsDice, ...pairedSynDice, 0.75);

        const scatterDice = {{
            x: pairedAntsDice,
            y: pairedSynDice,
            text: pairedDiceLabels,
            hoverinfo: 'text',
            mode: 'markers',
            type: 'scatter',
            name: 'Mindboggle Pairs',
            marker: {{
                size: 11,
                color: '#58a6ff',
                opacity: 0.85,
                line: {{ color: '#ffffff', width: 1.5 }}
            }}
        }};

        const lineDiceParity = {{
            x: [minDice, maxDice],
            y: [minDice, maxDice],
            mode: 'lines',
            type: 'scatter',
            name: 'Parity (y = x)',
            line: {{ dash: 'dash', color: '#8b949e', width: 2 }}
        }};

        Plotly.newPlot('diceScatterPlot', [scatterDice, lineDiceParity], {{
            title: {{ text: '<b>Cortical Accuracy: Syntx vs ANTs C++</b>', font: {{ color: '#ffffff', size: 15 }} }},
            xaxis: {{ title: 'ANTs C++ Symmetric Mean Dice', range: [minDice - 0.02, maxDice + 0.02], color: '#8b949e', gridcolor: '#21262d' }},
            yaxis: {{ title: 'Syntx PyTorch Symmetric Mean Dice', range: [minDice - 0.02, maxDice + 0.02], color: '#8b949e', gridcolor: '#21262d' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#c9d1d9' }},
            margin: {{ t: 50, b: 50, l: 60, r: 20 }},
            legend: {{ x: 0.05, y: 0.95, font: {{ color: '#c9d1d9' }} }}
        }}, {{ responsive: true }});

        // 2. X-Y Scatter: Syntx Compute Time vs ANTs Compute Time
        const pairedAntsTime = [];
        const pairedSynTime = [];
        const pairedTimeLabels = [];
        for (let i = 0; i < pairLabels.length; i++) {{
            if (antsTime[i] !== null && synTime[i] !== null) {{
                pairedAntsTime.push(antsTime[i]);
                pairedSynTime.push(synTime[i]);
                const sp = antsTime[i] / synTime[i];
                pairedTimeLabels.push(pairLabels[i] + '<br>Syntx GPU: ' + synTime[i] + 's<br>ANTs CPU: ' + antsTime[i] + 's<br>Speedup: ' + sp.toFixed(2) + 'x');
            }}
        }}

        const maxAntsTime = Math.max(...pairedAntsTime, 250);
        const maxSynTime = Math.max(...pairedSynTime, 100);

        const scatterTime = {{
            x: pairedAntsTime,
            y: pairedSynTime,
            text: pairedTimeLabels,
            hoverinfo: 'text',
            mode: 'markers',
            type: 'scatter',
            name: 'Mindboggle Pairs',
            marker: {{
                size: 11,
                color: '#bc8cff',
                opacity: 0.85,
                line: {{ color: '#ffffff', width: 1.5 }}
            }}
        }};

        const lineTimeParity = {{
            x: [0, maxAntsTime],
            y: [0, maxAntsTime],
            mode: 'lines',
            type: 'scatter',
            name: '1x (Parity)',
            line: {{ dash: 'dash', color: '#8b949e', width: 1.5 }}
        }};

        const lineTime2x = {{
            x: [0, maxAntsTime],
            y: [0, maxAntsTime * 0.5],
            mode: 'lines',
            type: 'scatter',
            name: '2x Speedup',
            line: {{ dash: 'dot', color: '#3fb950', width: 1.5 }}
        }};

        const lineTime3x = {{
            x: [0, maxAntsTime],
            y: [0, maxAntsTime * 0.333],
            mode: 'lines',
            type: 'scatter',
            name: '3x Speedup',
            line: {{ dash: 'dot', color: '#bc8cff', width: 1.5 }}
        }};

        Plotly.newPlot('timeScatterPlot', [scatterTime, lineTimeParity, lineTime2x, lineTime3x], {{
            title: {{ text: '<b>Compute Runtime: Syntx GPU vs ANTs CPU</b>', font: {{ color: '#ffffff', size: 15 }} }},
            xaxis: {{ title: 'ANTs C++ CPU Runtime (seconds)', range: [0, maxAntsTime + 20], color: '#8b949e', gridcolor: '#21262d' }},
            yaxis: {{ title: 'Syntx GPU Runtime (seconds)', range: [0, maxSynTime + 20], color: '#8b949e', gridcolor: '#21262d' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', color: '#c9d1d9' }},
            margin: {{ t: 50, b: 50, l: 60, r: 20 }},
            legend: {{ x: 0.05, y: 0.95, font: {{ color: '#c9d1d9' }} }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>
"""

    with open(out_html, "w") as f:
        f.write(html)
    print(f"Generated Side-by-Side HTML Report with X-Y Scatter Plots: {out_html} ({n_completed} pairs)", flush=True)

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "results/pairs_90/syn_mps"
    ants_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    out_file = sys.argv[3] if len(sys.argv) > 3 else "results/90pair_report.html"
    generate_html_report(res_dir, ants_dir, out_file)
