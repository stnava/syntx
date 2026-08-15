import os
import json
import glob
from datetime import datetime

def generate_report():
    syn_files = [f for f in glob.glob("results/pair_*_syn.json") if not f.endswith("_ants_syn.json")]
    ants_files = glob.glob("results/pair_*_ants_syn.json")
    
    syn_results = {}
    for f in syn_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            if data.get('status') == 'SUCCESS':
                syn_results[data.get('pair_idx')] = data
        except Exception:
            pass
            
    ants_results = {}
    for f in ants_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            if data.get('status') == 'SUCCESS':
                ants_results[data.get('pair_idx')] = data
        except Exception:
            pass

    completed = len(syn_results)
    total_pairs = 90
    
    # Compute paired stats
    paired_idx = set(syn_results.keys()).intersection(ants_results.keys())
    
    if completed == 0:
        mean_dice_syn = 0.0
        mean_fold_syn = 0.0
        mean_inv_syn = 0.0
    else:
        mean_dice_syn = sum(r.get('dice_sym', 0.0) for r in syn_results.values()) / completed
        mean_fold_syn = sum(r.get('folding_pct', 0.0) for r in syn_results.values()) / completed
        mean_inv_syn = sum(r.get('inverse_error_mean', 0.0) for r in syn_results.values()) / completed

    if len(ants_results) == 0:
        mean_dice_ants = 0.0
        mean_fold_ants = 0.0
        mean_inv_ants = 0.0
    else:
        mean_dice_ants = sum(r.get('dice_sym', 0.0) for r in ants_results.values()) / len(ants_results)
        mean_fold_ants = sum(r.get('folding_pct', 0.0) for r in ants_results.values()) / len(ants_results)
        mean_inv_ants = sum(r.get('inverse_error_mean', 0.0) for r in ants_results.values()) / len(ants_results)

    # Plot data
    pair_ids = [f"Pair {i}" for i in sorted(syn_results.keys())]
    
    syn_dice_sym = [syn_results[i].get('dice_sym', 0.0) for i in sorted(syn_results.keys())]
    ants_dice_sym = [ants_results.get(i, {}).get('dice_sym', None) for i in sorted(syn_results.keys())]
    
    syn_folds = [syn_results[i].get('folding_pct', 0.0) for i in sorted(syn_results.keys())]
    ants_folds = [ants_results.get(i, {}).get('folding_pct', None) for i in sorted(syn_results.keys())]

    syn_times = [syn_results[i].get('runtime_seconds', 0.0) for i in sorted(syn_results.keys())]
    ants_times = [ants_results.get(i, {}).get('runtime_seconds', None) for i in sorted(syn_results.keys())]
    
    syn_invs = [syn_results[i].get('inverse_error_mean', 0.0) for i in sorted(syn_results.keys())]
    ants_invs = [ants_results.get(i, {}).get('inverse_error_mean', None) for i in sorted(syn_results.keys())]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syntx Benchmark: PyTorch SyN vs ANTs C++</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --syntx-color: #3b82f6;
            --ants-color: #ef4444;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: var(--surface-color);
            padding: 50px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 8px;
        }}
        h1, h2, h3 {{ font-family: 'Crimson Pro', serif; color: #0f172a; }}
        h1 {{ font-size: 2.8rem; text-align: center; margin-bottom: 10px; font-weight: 700; }}
        .subtitle {{
            text-align: center; font-family: 'Inter', sans-serif;
            color: var(--text-muted); font-size: 1.1rem;
            margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px;
        }}
        .abstract {{
            font-style: italic; font-family: 'Crimson Pro', serif;
            font-size: 1.2rem; padding: 20px 40px;
            border-left: 4px solid var(--syntx-color);
            background-color: #f1f5f9; margin-bottom: 50px;
        }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background-color: var(--bg-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2.2rem; font-weight: 600; font-family: 'Inter', sans-serif; display: flex; justify-content: center; gap: 20px; }}
        .stat-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 500; margin-top: 5px; }}
        .color-syntx {{ color: var(--syntx-color); }}
        .color-ants {{ color: var(--ants-color); }}
        
        .plot-container {{ width: 100%; height: 500px; margin-bottom: 40px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        .plot-row {{ display: flex; gap: 20px; margin-bottom: 40px; flex-wrap: wrap; }}
        .plot-third {{ flex: 1; min-width: 300px; height: 500px; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.90rem; }}
        th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: var(--bg-color); font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; }}
        tr:hover td {{ background-color: #f8fafc; }}
        .progress-container {{ margin-bottom: 40px; }}
        .progress-bar {{ height: 6px; background-color: var(--border-color); border-radius: 3px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: var(--syntx-color); width: {(completed/total_pairs)*100}%; transition: width 1s ease; }}
        .progress-text {{ text-align: right; font-size: 0.85rem; color: var(--text-muted); margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparative SyN Registration Benchmark</h1>
        <div class="subtitle">Syntx PyTorch GPU vs. ANTs C++ CPU | Mindboggle-101</div>

        <div class="progress-container">
            <div class="progress-bar"><div class="progress-fill"></div></div>
            <div class="progress-text">Evaluation Progress: {completed} / {total_pairs} Pairs</div>
        </div>

        <div class="abstract">
            <strong>Abstract:</strong> This real-time whitepaper evaluates the strict numerical parity and spatial performance of the <i>syntx</i> native PyTorch Eulerian SyN formulation against the gold-standard ANTs C++ implementation. Both algorithms are evaluated systematically on the Mindboggle dataset, measuring symmetric DKT cortical overlap (Dice), topological folding, maximum inverse identity error, and execution speed.
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_dice_syn:.4f}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_dice_ants:.4f}</span>
                </div>
                <div class="stat-label">Mean Symmetric Dice (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_fold_syn:.4f}%</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_fold_ants:.4f}%</span>
                </div>
                <div class="stat-label">Mean Grid Folding (Syntx | ANTs)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">
                    <span class="color-syntx">{mean_inv_syn:.4f}</span>
                    <span style="color: #cbd5e1;">|</span>
                    <span class="color-ants">{mean_inv_ants:.4f}</span>
                </div>
                <div class="stat-label">Mean Inverse Error (mm)</div>
            </div>
        </div>

        <h2>I. Volumetric Overlap Analysis</h2>
        <p>Comparison of Symmetric Mean Dice distributions between the two implementations.</p>
        <div class="plot-container" id="diceBoxplot"></div>

        <h2>II. Performance & Topology Trade-offs</h2>
        <div class="plot-row">
            <div class="plot-third" id="foldScatter"></div>
            <div class="plot-third" id="invScatter"></div>
            <div class="plot-third" id="timeScatter"></div>
        </div>

        <h2>III. Paired Raw Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Pair ID</th>
                    <th>Syntx Dice</th>
                    <th>ANTs Dice</th>
                    <th>Syntx Fold</th>
                    <th>ANTs Fold</th>
                    <th>Syntx Mean Inv</th>
                    <th>ANTs Mean Inv</th>
                    <th>Syntx Time</th>
                    <th>ANTs Time</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx in sorted(syn_results.keys()):
        s_res = syn_results[idx]
        a_res = ants_results.get(idx, {})
        
        s_dice = f"{s_res.get('dice_sym', 0.0):.4f}"
        a_dice = f"{a_res.get('dice_sym', 0.0):.4f}" if a_res else "-"
        
        s_fold = f"{s_res.get('folding_pct', 0.0):.4f}%"
        a_fold = f"{a_res.get('folding_pct', 0.0):.4f}%" if a_res else "-"
        
        s_inv = f"{s_res.get('inverse_error_mean', 0.0):.4f}"
        a_inv = f"{a_res.get('inverse_error_mean', 0.0):.4f}" if a_res else "-"
        
        s_time = f"{s_res.get('runtime_seconds', 0.0):.1f}s"
        a_time = f"{a_res.get('runtime_seconds', 0.0):.1f}s" if a_res else "-"
        
        html += f"""
                <tr>
                    <td style="font-family: monospace; font-weight: 600;">#{idx:03d}</td>
                    <td class="color-syntx" style="font-weight: 600;">{s_dice}</td>
                    <td class="color-ants" style="font-weight: 600;">{a_dice}</td>
                    <td>{s_fold}</td>
                    <td>{a_fold}</td>
                    <td>{s_inv}</td>
                    <td>{a_inv}</td>
                    <td>{s_time}</td>
                    <td>{a_time}</td>
                </tr>"""

    html += f"""
            </tbody>
        </table>
    </div>

    <script>
        const pairIds = {json.dumps(pair_ids)};
        const synDice = {json.dumps(syn_dice_sym)};
        const antsDice = {json.dumps(ants_dice_sym)};
        const synFolds = {json.dumps(syn_folds)};
        const antsFolds = {json.dumps(ants_folds)};
        const synTimes = {json.dumps(syn_times)};
        const antsTimes = {json.dumps(ants_times)};
        const synInvs = {json.dumps(syn_invs)};
        const antsInvs = {json.dumps(ants_invs)};

        // 1. Boxplot (Dice)
        const traceSyn = {{ y: synDice, type: 'box', name: 'Syntx PyTorch', marker: {{color: '#3b82f6'}}, boxpoints: 'all', jitter: 0.3 }};
        const traceAnts = {{ y: antsDice, type: 'box', name: 'ANTs C++', marker: {{color: '#ef4444'}}, boxpoints: 'all', jitter: 0.3 }};
        
        Plotly.newPlot('diceBoxplot', [traceSyn, traceAnts], {{
            title: 'Symmetric DKT31 Dice Score Distributions',
            yaxis: {{ title: 'Dice Score', zeroline: false }},
            boxmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 2. Scatter (Dice vs Fold)
        const scatterSynFold = {{ x: synFolds, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsFold = {{ x: antsFolds, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('foldScatter', [scatterSynFold, scatterAntsFold], {{
            title: 'Dice vs Topology Destruction',
            xaxis: {{ title: 'Grid Folding % (det J <= 0)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});
        
        // 3. Scatter (Dice vs Inverse Error)
        const scatterSynInv = {{ x: synInvs, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsInv = {{ x: antsInvs, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};
        
        Plotly.newPlot('invScatter', [scatterSynInv, scatterAntsInv], {{
            title: 'Dice vs Mean Inverse Error',
            xaxis: {{ title: 'Mean Inverse Error (mm)', zeroline: false }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

        // 4. Scatter (Dice vs Time)
        const scatterSynTime = {{ x: synTimes, y: synDice, name: 'Syntx', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#3b82f6', opacity: 0.7 }} }};
        const scatterAntsTime = {{ x: antsTimes, y: antsDice, name: 'ANTs', text: pairIds, mode: 'markers', type: 'scatter', marker: {{ size: 8, color: '#ef4444', opacity: 0.7 }} }};

        Plotly.newPlot('timeScatter', [scatterSynTime, scatterAntsTime], {{
            title: 'Runtime Performance',
            xaxis: {{ title: 'Compute Time (seconds)' }},
            yaxis: {{ title: 'Symmetric Mean Dice' }},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif' }}
        }}, {{responsive: true}});

    </script>
</body>
</html>
"""
    
    with open("results/90pair_report.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    generate_report()
    print("Comparative report generated at results/90pair_report.html")
