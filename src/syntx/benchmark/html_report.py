"""
syntx.benchmark.html_report — Live HTML Dashboard for Population Registration Benchmarks
========================================================================================

Generates an auto-refreshing, self-contained HTML dashboard for tracking multi-model
population registration progress, per-pair accuracy, and topological metrics in real-time.
"""

import os
import json
import time
import glob
import statistics
from typing import Dict, List, Any, Optional


METHOD_METADATA = {
    "syn": {"name": "Eulerian Sobolev SyN", "badge": "#0284c7"},
    "gaussian": {"name": "Eulerian Gaussian SyN", "badge": "#ec4899"},
    "syngs": {"name": "Geodesic Shooting (SyNGS)", "badge": "#8b5cf6"},
    "tvf": {"name": "Time-Varying Velocity Field (TVF)", "badge": "#059669"},
    "greedy": {"name": "Compositive Greedy (LDdMM)", "badge": "#d97706"},
    "ants_syn": {"name": "ANTs C++ SyN Baseline", "badge": "#64748b"},
}


def load_model_results_from_disk(results_dir: str, model: str, device: str) -> List[Dict[str, Any]]:
    """Loads all completed pair JSON files for a given model from disk."""
    m_dir = os.path.join(results_dir, f"{model}_{device}")
    if not os.path.isdir(m_dir):
        return []
    
    results = []
    pattern = os.path.join(m_dir, f"pair_*_{model}.json")
    for filepath in sorted(glob.glob(pattern)):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if data.get("status") == "SUCCESS":
                results.append(data)
        except Exception:
            pass
    return results


def compute_model_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes summary statistics across completed runs for a single model."""
    if not results:
        return {
            "count": 0,
            "mean_dice": None, "std_dice": None, "med_dice": None,
            "mean_fold": None, "min_jac": None, "mean_time": None,
            "win_rate": None,
        }
    
    dices = [r["dice_sym"] for r in results if "dice_sym" in r and r["dice_sym"] is not None and str(r["dice_sym"]) != "nan"]
    folds = [r["folding_pct"] for r in results if "folding_pct" in r and r["folding_pct"] is not None and str(r["folding_pct"]) != "nan"]
    jacs = [r["min_jacobian"] for r in results if "min_jacobian" in r and r["min_jacobian"] is not None and str(r["min_jacobian"]) != "nan"]
    times = [r["runtime_seconds"] for r in results if "runtime_seconds" in r and r["runtime_seconds"] is not None and str(r["runtime_seconds"]) != "nan"]
    wins = [r.get("win", False) for r in results]

    return {
        "count": len(results),
        "mean_dice": statistics.mean(dices) if dices else 0.0,
        "std_dice": statistics.stdev(dices) if len(dices) > 1 else 0.0,
        "med_dice": statistics.median(dices) if dices else 0.0,
        "mean_fold": statistics.mean(folds) if folds else 0.0,
        "min_jac": min(jacs) if jacs else 1.0,
        "mean_time": statistics.mean(times) if times else 0.0,
        "total_time": sum(times) if times else 0.0,
        "win_rate": (sum(1 for w in wins if w) / len(wins) * 100.0) if wins else 0.0,
    }


def generate_live_html_report(
    results_dir: str = "results",
    device: str = "mps",
    models: Optional[List[str]] = None,
    total_pairs: int = 90,
    out_html: str = "results/multimodel_live_comparison.html",
    refresh_seconds: int = 10,
) -> str:
    """Generates an auto-refreshing HTML benchmark dashboard and saves it to disk."""
    if models is None:
        models = ["syn", "gaussian", "syngs", "tvf", "greedy"]

    # Load results per model
    model_data: Dict[str, List[Dict[str, Any]]] = {}
    model_stats: Dict[str, Dict[str, Any]] = {}
    
    for m in models:
        res = load_model_results_from_disk(results_dir, m, device)
        model_data[m] = res
        model_stats[m] = compute_model_stats(res)

    # Gather per-pair data grouped by pair_idx
    pairs_map: Dict[int, Dict[str, Any]] = {}
    for m, res_list in model_data.items():
        for r in res_list:
            p_idx = r.get("pair_idx")
            if p_idx is None:
                continue
            if p_idx not in pairs_map:
                pairs_map[p_idx] = {
                    "pair_idx": p_idx,
                    "fixed_id": r.get("fixed_id", f"Pair {p_idx}"),
                    "moving_id": r.get("moving_id", ""),
                    "cohort_type": r.get("cohort_type", ""),
                    "ants_baseline": r.get("ants_baseline", {}),
                    "models": {},
                }
            pairs_map[p_idx]["models"][m] = r

    all_pair_indices = sorted(pairs_map.keys())
    fully_completed_pairs = [
        p for p in all_pair_indices
        if all(m in pairs_map[p]["models"] for m in models)
    ]
    
    pct_complete = (len(fully_completed_pairs) / max(1, total_pairs)) * 100.0
    now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="{refresh_seconds}">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>syntx Multi-Model Benchmark Dashboard ({device.upper()})</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --text: #1e293b;
            --muted: #64748b;
            --border: #e2e8f0;
            --accent: #0284c7;
            --success: #059669;
            --success-bg: #dcfce7;
            --amber: #d97706;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            padding: 36px 40px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .header .badge {{
            display: inline-block;
            font-size: 13px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 9999px;
            background: #e0f2fe;
            color: #0369a1;
        }}
        .timer {{
            font-size: 13px;
            color: var(--muted);
            background: #f1f5f9;
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .progress-box {{
            margin: 20px 0 28px 0;
        }}
        .progress-header {{
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .progress-bar-bg {{
            width: 100%;
            height: 14px;
            background: #e2e8f0;
            border-radius: 9999px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            width: {pct_complete:.1f}%;
            background: linear-gradient(90deg, #0284c7, #059669);
            border-radius: 9999px;
            transition: width 0.4s ease;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}
        .card-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
        }}
        .card-val {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 6px;
        }}
        .card-sub {{
            font-size: 12px;
            color: var(--muted);
            margin-top: 4px;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin: 28px 0 16px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-bottom: 24px;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            text-align: left;
            padding: 12px 14px;
            border-top: 1px solid var(--border);
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 11px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{
            background: #f8fafc;
        }}
        .best-val {{
            background-color: var(--success-bg);
            color: var(--success);
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .pill {{
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 9999px;
            text-transform: uppercase;
        }}
        .footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>syntx Population Benchmark <span class="badge">{device.upper()}</span></h1>
                <p style="font-size: 13px; color: var(--muted); margin-top: 4px;">
                    Standardized cc2 Metric &bull; [100, 100, 20] Iterations &bull; Canonical pt7 Affine
                </p>
            </div>
            <div class="timer">
                <span>🔄 Auto-refreshing in <strong id="countdown">{refresh_seconds}</strong>s</span> &bull; 
                <span>Last Updated: {now_utc}</span>
            </div>
        </div>

        <div class="progress-box">
            <div class="progress-header">
                <span>Cohort Progress: {len(fully_completed_pairs)} of {total_pairs} Pairs Completed</span>
                <span>{pct_complete:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill"></div>
            </div>
        </div>

        <div class="card-grid">
            <div class="card">
                <div class="card-label">Active Sequence</div>
                <div class="card-val" style="font-size: 20px;">Randomized (Seed 42)</div>
                <div class="card-sub">{len(all_pair_indices)} distinct pairs registered</div>
            </div>
"""

    for m in models:
        st = model_stats.get(m, {})
        meta = METHOD_METADATA.get(m, {"name": m.upper(), "badge": "#0284c7"})
        dice_disp = f"{st['mean_dice']:.4f}" if st.get("mean_dice") is not None else "Pending"
        cnt = st.get("count", 0)
        html += f"""
            <div class="card">
                <div class="card-label">{meta['name']}</div>
                <div class="card-val" style="color: {meta['badge']};">
                    {dice_disp}
                </div>
                <div class="card-sub">Completed: {cnt}/{total_pairs} pairs</div>
            </div>
"""

    html += """
        </div>

        <h2>Method Comparison Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Model Description</th>
                    <th>Completed</th>
                    <th>Mean Symmetric Dice (&plusmn; std)</th>
                    <th>Median Dice</th>
                    <th>Mean Folding %</th>
                    <th>Min |J|</th>
                    <th>Mean Time (s)</th>
                    <th>Win Rate vs ANTs</th>
                </tr>
            </thead>
            <tbody>
"""

    for m in models:
        st = model_stats[m]
        meta = METHOD_METADATA.get(m, {"name": m.upper(), "badge": "#64748b"})
        if st["count"] > 0:
            html += f"""
                <tr>
                    <td><strong style="color: {meta['badge']};">{m.upper()}</strong></td>
                    <td>{meta['name']}</td>
                    <td><strong>{st['count']}</strong> / {total_pairs}</td>
                    <td><strong>{st['mean_dice']:.4f}</strong> &plusmn; {st['std_dice']:.4f}</td>
                    <td>{st['med_dice']:.4f}</td>
                    <td>{st['mean_fold']:.4f}%</td>
                    <td>{st['min_jac']:+.4f}</td>
                    <td>{st['mean_time']:.1f}s</td>
                    <td><span class="pill" style="background: #dcfce7; color: #166534;">{st['win_rate']:.1f}%</span></td>
                </tr>
            """
        else:
            html += f"""
                <tr>
                    <td><strong style="color: {meta['badge']};">{m.upper()}</strong></td>
                    <td>{meta['name']}</td>
                    <td>0 / {total_pairs}</td>
                    <td>Pending</td>
                    <td>Pending</td>
                    <td>Pending</td>
                    <td>Pending</td>
                    <td>Pending</td>
                    <td>Pending</td>
                </tr>
            """

    html += """
            </tbody>
        </table>

        <h2>Per-Pair Head-to-Head Comparison (Sorted by Pair Index)</h2>
        <table>
            <thead>
                <tr>
                    <th>Pair</th>
                    <th>Subject Pair</th>
                    <th>Type</th>
                    <th>ANTs Baseline</th>
"""
    for m in models:
        meta = METHOD_METADATA.get(m, {"name": m.upper()})
        html += f"                    <th>{meta['name']} Dice (Fold)</th>\n"
    html += """                    <th>Best Performer</th>
                </tr>
            </thead>
            <tbody>
"""

    for p_idx in all_pair_indices:
        p_info = pairs_map[p_idx]
        fixed_id = p_info["fixed_id"]
        moving_id = p_info["moving_id"]
        c_type = p_info["cohort_type"] or ("intra" if p_idx < 40 else "inter")
        ants_base = p_info["ants_baseline"].get("dice_sym")
        ants_str = f"{ants_base:.4f}" if ants_base is not None else "&mdash;"

        # Find best model
        eval_scores = {}
        for m in models:
            if m in p_info["models"]:
                d = p_info["models"][m].get("dice_sym")
                if d is not None and str(d) != "nan":
                    eval_scores[m] = d

        best_m = max(eval_scores, key=eval_scores.get) if eval_scores else None

        def fmt_cell(m_name):
            if m_name not in p_info["models"]:
                return "<span style='color: var(--muted);'>Pending</span>"
            res = p_info["models"][m_name]
            d = res.get("dice_sym", 0.0)
            f = res.get("folding_pct", 0.0)
            is_best = (m_name == best_m)
            dice_cls = "class='best-val'" if is_best else ""
            return f"<span {dice_cls} title='Dice: {d:.6f} | Fold: {f:.5f}%'>{d:.4f}</span> <span style='font-size: 11px; color: var(--muted);'>({f:.3f}%)</span>"

        model_cells = "".join([f"                <td>{fmt_cell(m)}</td>\n" for m in models])
        html += f"""
            <tr>
                <td><strong>#{p_idx:03d}</strong></td>
                <td><code>{fixed_id}</code> &rarr; <code>{moving_id}</code></td>
                <td><span class="pill" style="background: {'#fef3c7' if c_type == 'inter' else '#f1f5f9'}; color: {'#92400e' if c_type == 'inter' else '#475569'};">{c_type}</span></td>
                <td>{ants_str}</td>
{model_cells}                <td><strong>{best_m.upper() if best_m else '&mdash;'}</strong></td>
            </tr>
        """

    html += f"""
            </tbody>
        </table>

        <div class="footer">
            <span>syntx v5.4.9 &bull; Standardized Diffeomorphic Evaluation Protocol</span>
            <span>Single source of truth: <code>src/syntx/benchmark/config.py</code></span>
        </div>
    </div>

    <script>
        let timeLeft = {refresh_seconds};
        const el = document.getElementById('countdown');
        setInterval(() => {{
            timeLeft--;
            if (timeLeft < 0) timeLeft = {refresh_seconds};
            if (el) el.textContent = timeLeft;
        }}, 1000);
    </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    temp_html = out_html + ".tmp"
    with open(temp_html, "w") as f:
        f.write(html)
    os.replace(temp_html, out_html)
    return out_html
