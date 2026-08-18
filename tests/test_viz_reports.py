import os
import random
from syntx.viz.reports import create_benchmark_report

def test_benchmark_report_generation():
    """
    Tests the generation of the interactive comparative benchmark HTML report
    using on-the-fly simulated data.
    """
    total_pairs = 20
    syn_results = {}
    ants_results = {}
    
    # Simulate data where Syntx is slightly better on average
    for i in range(total_pairs):
        base_dice = random.uniform(0.50, 0.70)
        syn_results[i] = {
            'pair_idx': i,
            'status': 'SUCCESS',
            'dice_sym': base_dice + random.uniform(0.01, 0.03),  # Syntx slightly better
            'folding_pct': random.uniform(0.00, 0.05),
            'inverse_error_mean': random.uniform(0.01, 0.05),
            'runtime_seconds': random.uniform(20.0, 40.0)
        }
        ants_results[i] = {
            'pair_idx': i,
            'status': 'SUCCESS',
            'dice_sym': base_dice,
            'folding_pct': random.uniform(0.00, 0.08),
            'inverse_error_mean': random.uniform(0.02, 0.06),
            'runtime_seconds': random.uniform(40.0, 90.0)  # ANTs slower
        }

    output_html = "test_simulated_report.html"
    
    # Clean up before test
    if os.path.exists(output_html):
        os.remove(output_html)

    # Generate report
    result_path = create_benchmark_report(
        syn_results=syn_results,
        ants_results=ants_results,
        total_pairs=total_pairs,
        output_html=output_html
    )

    # Assert file was created and contains expected plot elements
    assert os.path.exists(result_path), f"Report file {result_path} was not created."
    
    with open(result_path, "r") as f:
        html_content = f.read()
        
    # Verify basic plots are included
    assert "Paired T-Test p-value" in html_content
    assert "diceBoxplot" in html_content
    assert "pairedScatter" in html_content
    
    # Catch f-string un-interpolated variables (e.g. {s_dice})
    assert "{s_dice}" not in html_content, "Found un-interpolated {s_dice} variable in HTML"
    assert "{a_dice}" not in html_content, "Found un-interpolated {a_dice} variable in HTML"
    assert "{s_fold}" not in html_content, "Found un-interpolated {s_fold} variable in HTML"
    
    # Catch JS object double-brace escaping bugs (e.g. {{ x: ... }})
    assert "{{ x:" not in html_content, "Found escaped double-brace {{ in JS payload, which breaks Plotly"
    assert "}}" not in html_content, "Found escaped double-brace }} in JS payload, which breaks Plotly"
    
    # Catch JSON dumps literal failures (e.g. {json.dumps(pair_ids)})
    assert "{json.dumps" not in html_content, "Found un-evaluated json.dumps in HTML"
    
    # Verify Javascript arrays were successfully populated
    import re
    assert re.search(r"const pairIds = \[\"Pair 0\"", html_content) is not None, "JS array pairIds was not generated correctly"
    assert re.search(r"const synDice = \[.*?\];", html_content) is not None, "JS array synDice was not generated correctly"
    
    if os.path.exists(output_html):
        os.remove(output_html)


def test_affine_benchmark_report_generation():
    """
    Tests generation of the interactive 90-pair affine registration benchmark report.
    """
    from syntx.viz.reports import create_affine_benchmark_report
    output_html = "test_affine_report.html"

    if os.path.exists(output_html):
        os.remove(output_html)

    result_path = create_affine_benchmark_report(
        summary_source="results/reproducible_90pair_master_summary.json",
        output_html=output_html
    )

    assert os.path.exists(result_path), f"Report file {result_path} was not created."

    with open(result_path, "r") as f:
        html_content = f.read()

    assert "Syntx Robust Affine" in html_content
    assert "progressionPlot" in html_content
    assert "boxPlot" in html_content
    assert "correlationPlot" in html_content
    assert "runtimePlot" in html_content
    assert "{mean_aff}" not in html_content
    assert "{{" not in html_content
    assert "}}" not in html_content

    if os.path.exists(output_html):
        os.remove(output_html)


if __name__ == "__main__":
    test_benchmark_report_generation()
    test_affine_benchmark_report_generation()
    print("All report tests passed successfully!")
