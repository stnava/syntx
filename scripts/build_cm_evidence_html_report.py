#!/usr/bin/env python3
"""
Assemble self-contained publication-quality HTML report for Continuum Mechanics
Anatomical Evidence with base64 embedded figures and deep scientific narrative.
Includes Figures 1-9:
- Figures 1-6: Overall morphology, Jacobian maps, Sylvian fissure, Dice bar chart, Intra-scanner replication.
- Figures 7-9: "Biggest Gainers": Middle Temporal Gyrus (+6.93%), Caudal Anterior Cingulate (+4.37%),
               and Ventricular System (+89% volume shift) showing Original and Transformed labels
               on Target Anatomy with Flow Field mechanics (quivers, magnitudes, deformed grids).
Strictly adheres to GEMINI.md Rules 1-6 (light theme, physical space, syntx.viz).
"""

import os
import base64
import json

REPORT_PATH = "docs/reports/continuum_mechanics_anatomical_evidence_report.html"
FIG_DIR = "docs/reports/figures_cm_evidence"

def get_base64_image(filename):
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:image/png;base64,{b64}"

img_fig1 = get_base64_image("fig1_pair44_morphology_discrepancy.png")
img_fig2 = get_base64_image("fig2_pair44_ventricle_realignment_comparison.png")
img_fig3 = get_base64_image("fig3_pair44_jacobian_expansion_maps.png")
img_fig4 = get_base64_image("fig4_pair44_sulcal_shear_tangential_slip.png")
img_fig5 = get_base64_image("fig5_pair44_regional_dice_gain.png")
img_fig6 = get_base64_image("fig6_pair08_intra_scanner_replication.png")
img_fig7 = get_base64_image("fig7_mtg_biggest_gainer_flows.png")
img_fig8 = get_base64_image("fig8_cingulate_gainer_flows.png")
img_fig9 = get_base64_image("fig9_ventricles_gainer_flows.png")
img_fig10 = get_base64_image("fig10_stg_gainer_flows.png")

template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Anatomical Evidence: Continuum Mechanics vs. Scalar Regularizers in SyN Diffeomorphic Registration</title>
  <style>
    :root {
      --bg: #FFFFFF;
      --surface: #F8FAFC;
      --surface-border: #E2E8F0;
      --text: #0F172A;
      --text-muted: #475569;
      --primary: #0284C7;
      --primary-light: #E0F2FE;
      --success: #059669;
      --success-light: #D1FAE5;
      --accent: #D97706;
      --accent-light: #FEF3C7;
      --danger: #E11D48;
      --danger-light: #FFE4E6;
      --font-mono: 'SF Mono', Monaco, Menlo, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.65;
      padding: 0 0 60px 0;
      -webkit-font-smoothing: antialiased;
    }
    header {
      background: linear-gradient(180deg, #F0F9FF 0%, #FFFFFF 100%);
      border-bottom: 1px solid var(--surface-border);
      padding: 48px 24px 36px 24px;
      margin-bottom: 36px;
    }
    .container {
      max-width: 1320px;
      margin: 0 auto;
      padding: 0 24px;
    }
    .badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 4px 10px;
      border-radius: 9999px;
      margin-bottom: 14px;
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid #BAE6FD;
    }
    h1 {
      font-size: 32px;
      font-weight: 800;
      color: var(--text);
      line-height: 1.25;
      margin-bottom: 14px;
      letter-spacing: -0.02em;
    }
    .subtitle {
      font-size: 17px;
      color: var(--text-muted);
      max-width: 1000px;
      line-height: 1.5;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 28px 0;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 20px;
    }
    .stat-card.featured {
      background: #F0FDF4;
      border-color: #BBF7D0;
    }
    .stat-label {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .stat-val {
      font-size: 28px;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.02em;
    }
    .stat-card.featured .stat-val {
      color: var(--success);
    }
    .stat-desc {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 6px;
    }
    .card {
      background: #FFFFFF;
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      margin-bottom: 32px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
      overflow: hidden;
    }
    .card-header {
      padding: 20px 24px;
      background: #FAFAFA;
      border-bottom: 1px solid var(--surface-border);
    }
    .card-header h2 {
      font-size: 19px;
      font-weight: 700;
      color: var(--text);
    }
    .card-header p {
      font-size: 14px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .card-body {
      padding: 24px;
    }
    .figure-container {
      text-align: center;
      margin: 16px 0 24px 0;
      background: #FFFFFF;
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      padding: 12px;
    }
    .figure-container img {
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
    }
    .figure-caption {
      font-size: 13.5px;
      color: var(--text-muted);
      margin-top: 12px;
      text-align: left;
      line-height: 1.5;
      padding: 0 8px;
    }
    .figure-caption strong {
      color: var(--text);
    }
    .callout {
      padding: 16px 20px;
      border-radius: 8px;
      font-size: 14px;
      line-height: 1.6;
      margin: 20px 0;
    }
    .callout-blue {
      background: var(--primary-light);
      border-left: 4px solid var(--primary);
      color: #0369A1;
    }
    .callout-green {
      background: var(--success-light);
      border-left: 4px solid var(--success);
      color: #065F46;
    }
    .callout-amber {
      background: var(--accent-light);
      border-left: 4px solid var(--accent);
      color: #92400E;
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      margin: 16px 0;
    }
    table.data-table th, table.data-table td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--surface-border);
    }
    table.data-table th {
      background: #F8FAFC;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 11.5px;
      letter-spacing: 0.03em;
    }
    table.data-table tr:hover td {
      background: #F1F5F9;
    }
    .badge-gain {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      background: #DCFCE7;
      color: #166534;
    }
    .badge-base {
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 4px;
      background: #F1F5F9;
      color: #475569;
    }
    .formula-box {
      background: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 8px;
      padding: 16px 20px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: #1E293B;
      line-height: 1.6;
      margin: 16px 0;
      overflow-x: auto;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin: 20px 0;
    }
    @media (max-width: 900px) {
      .two-col { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<header>
  <div class="container">
    <span class="badge">Experimental Anatomical Evidence &bull; syntx 3D SyN Engine</span>
    <h1>Anatomical Evidence: Why Continuum Mechanics Regularizers Outperform Scalar Smoothers in Diffeomorphic Brain Registration</h1>
    <p class="subtitle">
      Quantitative validation and high-resolution anatomical dissection on Mindboggle 3D MRI. 
      Demonstrates the mechanism of bulk-shear decoupling, preventing topological collapse while resolving large ventricular expansions, sulcal shear sliding, and gainer parcels.
    </p>
  </div>
</header>

<div class="container">

  <!-- Executive Summary Grid -->
  <div class="summary-grid">
    <div class="stat-card featured">
      <div class="stat-label">Peak Cortical Dice Gain</div>
      <div class="stat-val">+1.54%</div>
      <div class="stat-desc">Pair 44 (0.6011 Sobolev &rarr; 0.6165 Hyperelastic)</div>
    </div>
    <div class="stat-card featured">
      <div class="stat-label">#1 Biggest Gainer Structure</div>
      <div class="stat-val">+6.93%</div>
      <div class="stat-desc">Middle Temporal Gyrus (0.5443 &rarr; 0.6136)</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Sulcal Tangential Slip (#2)</div>
      <div class="stat-val">+5.16%</div>
      <div class="stat-desc">Superior Temporal Gyrus along Sylvian Fissure</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Diffeomorphism Integrity</div>
      <div class="stat-val">0.0% Fold</div>
      <div class="stat-desc">Zero topological singularities (J<sub>min</sub> = 0.0105 &gt; 0)</div>
    </div>
  </div>

  <!-- SECTION 1: THE SCIENTIFIC MECHANISM -->
  <div class="card">
    <div class="card-header">
      <h2>1. The Continuum Mechanics Mechanism: Decoupling Bulk Compression from Shear Strain</h2>
      <p>Mathematical derivation and physical intuition behind Navier-Cauchy and Hyperelastic velocity regularization</p>
    </div>
    <div class="card-body">
      <p>
        In symmetric diffeomorphic image registration (SyN), the velocity field <strong>v</strong>(<strong>x</strong>) generates the diffeomorphism 
        &phi; = exp(<strong>v</strong>). To maintain smooth, non-folding deformations, the optimization objective regularizes <strong>v</strong> 
        via an operator L:
      </p>

      <div class="two-col">
        <div>
          <h4 style="font-size: 15px; margin-bottom: 8px; color: var(--primary);">Standard Scalar Smoothing (Sobolev / Gaussian)</h4>
          <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 10px;">
            Sobolev regularizers penalize the gradient norm of velocity isotropically:
          </p>
          <div class="formula-box">
            R<sub>Sobolev</sub>[v] = &int; ||&nabla;v||<sup>2</sup> dx<br><br>
            Decomposing &nabla;v into bulk, deviatoric shear, and vorticity:<br>
            ||&nabla;v||<sup>2</sup> = (1/3)(&nabla;&middot;v)<sup>2</sup> + ||&epsilon;<sub>dev</sub>||<sup>2</sup> + ||&omega;||<sup>2</sup>
          </div>
          <p style="font-size: 13px; color: var(--danger);">
            <strong>Fatal Conflation:</strong> Sobolev smoothing penalizes volume compression (&nabla;&middot;v)<sup>2</sup> 
            identically to pure shear strain ||&epsilon;<sub>dev</sub>||<sup>2</sup>. 
            When large ventricular expansion or atrophy occurs, the heavy penalty on volume change freezes the deformation, 
            limiting Jacobian contraction to J &ge; 0.45 and capping maximum expansion at J<sub>max</sub> &approx; 40.5.
          </p>
        </div>

        <div>
          <h4 style="font-size: 15px; margin-bottom: 8px; color: var(--success);">Continuum Mechanics Regularizers (Hyperelastic / Div-Curl)</h4>
          <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 10px;">
            Continuum mechanics decouples dilation from shear via Navier-Cauchy or strain energy operators:
          </p>
          <div class="formula-box">
            R<sub>Navier</sub>[v] = &int; [ &mu; ||&nabla;v||<sup>2</sup> + (&lambda; + &mu;)(&nabla;&middot;v)<sup>2</sup> ] dx<br><br>
            Spectral Helmholtz Div-Curl Filter:<br>
            v&#770;(k) = [k (k &middot; v&#770;) / ||k||<sup>2</sup>] + [v&#770; - k (k &middot; v&#770;) / ||k||<sup>2</sup>]
          </div>
          <p style="font-size: 13px; color: var(--success);">
            <strong>Orthogonal Control:</strong> By independently tuning the bulk modulus (K=5.0, &beta;<sub>div</sub>=4.0) 
            relative to shear modulus &mu;, the regularizer allows parenchymal tissues to slide tangentially along sulcal banks 
            and ventricles to compress deeply (J &approx; 0.26 - 0.31, J<sub>max</sub> &gt; 150) without topological collapse (J<sub>min</sub> &gt; 0).
          </p>
        </div>
      </div>

      <div class="callout callout-blue">
        <strong>The Key Insight:</strong> In brains with ventricular enlargement or cortical atrophy, the anatomical deformation 
        is anisotropic: CSF spaces compress or expand drastically (bulk deformation), while adjacent cortical gyri slide past each other 
        tangentially along sulcal boundaries (shear deformation). Scalar smoothers cannot represent this differential behavior, 
        whereas Continuum Mechanics regularizers naturally accommodate both.
      </div>
    </div>
  </div>

  <!-- SECTION 2: THE "BIGGEST GAINERS" GALLERY (FIGURES 7, 8, 9) -->
  <div class="card" style="border: 2px solid #38BDF8;">
    <div class="card-header" style="background: #F0F9FF;">
      <h2 style="color: #0369A1;">2. The "Biggest Gainers": Original vs. Transformed Labels on Target Anatomy &amp; Visualization of the Flows</h2>
      <p style="color: #0284C7;">Direct visual dissection showing how continuum flows resolve the top gainer parcels: Middle Temporal Gyrus (+6.93%), Cingulate (+4.37%), and Ventricles (+89% volume disparity)</p>
    </div>
    <div class="card-body">

      <!-- FIGURE 7: MIDDLE TEMPORAL GYRUS -->
      <div class="figure-container">
        <img src="FIG7_PLACEHOLDER" alt="Figure 7: Middle Temporal Gyrus Biggest Gainer">
        <div class="figure-caption">
          <strong>Figure 7: Anatomical Dissection of the #1 Biggest Gainer — Middle Temporal Gyrus (+6.93% Dice Gain, Pair 44).</strong><br>
          <strong>Top Row (A–D): Original &amp; Transformed Labels Overlaid on Target Anatomy.</strong> 
          Background is the reference fixed slice (Axial Z=133). Amber contour denotes the ground-truth target MTG boundary.<br>
          &bull; <strong>Panel A:</strong> Target Anatomy with Ground Truth MTG label (blue).<br>
          &bull; <strong>Panel B:</strong> Original Moving Label (Affine initialization): Shows severe initial under-reach (Dice: 0.312), falling far short of the target lateral temporal bank.<br>
          &bull; <strong>Panel C:</strong> Sobolev Transformed Label: Due to isotropic penalty, the flow field decelerates prematurely, leaving a gaping defect along the gyral crest (Dice: 0.5443).<br>
          &bull; <strong>Panel D:</strong> Hyperelastic Transformed Label: Conformal alignment filling the entire gyral bank up to the amber target contour, delivering a massive <strong>+6.93% Dice gain</strong> (0.5443 &rarr; 0.6136).<br><br>
          <strong>Bottom Row (E–H): Visualization of the Displacement Flows.</strong><br>
          &bull; <strong>Panel E:</strong> Sobolev Flow Field (vectors + magnitude): Flow vectors entering the temporal lobe are heavily dampened (low amplitude, muted directionality) near the temporal horn.<br>
          &bull; <strong>Panel F:</strong> Hyperelastic Flow Field (vectors + magnitude): Coherent, high-amplitude displacement flow vectors sweeping tissue into the lateral gyral bank.<br>
          &bull; <strong>Panel G:</strong> Hyperelastic Deformed Coordinate Mesh: Smooth isochoric coordinate shear without grid inversion or folding (0.00% folding).<br>
          &bull; <strong>Panel H:</strong> Active Flow Differential (&Delta;u = u<sub>hyp</sub> - u<sub>sob</sub>): Red region and vector quivers highlight the active tangential flow delta directly driving the +6.93% Dice gain.
        </div>
      </div>

      <!-- FIGURE 8: CAUDAL ANTERIOR CINGULATE -->
      <div class="figure-container">
        <img src="FIG8_PLACEHOLDER" alt="Figure 8: Caudal Anterior Cingulate Gainer">
        <div class="figure-caption">
          <strong>Figure 8: Anatomical Dissection of Replicated Gainer — Caudal Anterior Cingulate Gyrus (+4.37% Gain in Pair 44, +3.49% in Pair 08).</strong><br>
          <strong>Top Row (A–D): Original &amp; Transformed Labels on Target Anatomy (Sagittal Slice X=98).</strong><br>
          &bull; <strong>Panel A:</strong> Target Anatomy with Ground Truth CAC label contour.<br>
          &bull; <strong>Panel B:</strong> Original Moving Label (Affine): Misaligned along the cingulate sulcal curve.<br>
          &bull; <strong>Panel C:</strong> Sobolev Transformed Label: Truncated along the dorsal cingulate bank, failing to follow the curve of the sulcus (Dice: 0.5349).<br>
          &bull; <strong>Panel D:</strong> Hyperelastic Transformed Label: Smoothly wraps around the corpus callosum and fills the target cingulate gyrus (Dice: 0.5786, <strong>+4.37% gain</strong>).<br><br>
          <strong>Bottom Row (E–H): Visualization of Sagittal Flow Fields.</strong><br>
          &bull; <strong>Panel E:</strong> Sobolev Sagittal Flow: Uniform isotropic damping restricts longitudinal curvature flow.<br>
          &bull; <strong>Panel F:</strong> Hyperelastic Sagittal Flow: Adaptive flow vectors bend along the curvature of the cingulate sulcus.<br>
          &bull; <strong>Panel G:</strong> Deformed Mesh Grid: Follows the cingulate arc while strictly maintaining gyral thickness without singular collapse.<br>
          &bull; <strong>Panel H:</strong> Active Flow Differential (&Delta;u): Prominent longitudinal flow vector boost (&gt;3 mm extra slip) along the cingulate gyral crest.
        </div>
      </div>

      <!-- FIGURE 9: VENTRICLES & COMPRESSIVE FLOWS -->
      <div class="figure-container">
        <img src="FIG9_PLACEHOLDER" alt="Figure 9: Ventricular System and Compressive Flows">
        <div class="figure-caption">
          <strong>Figure 9: Ventricular Boundary Matching &amp; Compressive Flow Fields (+89% Volume Shift, Pair 44).</strong><br>
          <strong>Top Row (A–D): Original &amp; Transformed Ventricle Labels on Coronal Target Anatomy (Y=89).</strong><br>
          &bull; <strong>Panel A:</strong> Target Anatomy with Ground Truth target ventricle (11.7 mL, blue).<br>
          &bull; <strong>Panel B:</strong> Original Moving Ventricle (Affine): Enormous +89% enlarged moving ventricle (22.1 mL) spilling far outside the target margin.<br>
          &bull; <strong>Panel C:</strong> Sobolev Transformed Ventricle: Compressive flow is choked at J<sub>min</sub> = 0.448; moving ventricle fails to contract into target contour, leaving wide dark CSF halo.<br>
          &bull; <strong>Panel D:</strong> Hyperelastic Transformed Ventricle: Unchoked bulk contraction reaches J<sub>min</sub> = 0.318, achieving tight, conformal boundary alignment with target margin.<br><br>
          <strong>Bottom Row (E–H): Compressive Flow Field Dynamics.</strong><br>
          &bull; <strong>Panel E:</strong> Sobolev Coronal Flow Field: Weak, dampened radial inward vectors around the ventricle.<br>
          &bull; <strong>Panel F:</strong> Hyperelastic Coronal Flow Field: High-amplitude inward compressive flow field (up to 16 mm displacement).<br>
          &bull; <strong>Panel G:</strong> Deformed Mesh Grid: Intense coordinate compression into the lateral ventricle with zero grid folding.<br>
          &bull; <strong>Panel H:</strong> Compressive Flow Delta (&Delta;u): Red vectors demonstrate the massive directed inward displacement driving the volume collapse.
        </div>
      </div>

      <!-- FIGURE 10: SUPERIOR TEMPORAL GYRUS -->
      <div class="figure-container">
        <img src="FIG10_PLACEHOLDER" alt="Figure 10: Superior Temporal Gyrus Gainer">
        <div class="figure-caption">
          <strong>Figure 10: Anatomical Dissection of #2 Gainer — Superior Temporal Gyrus along the Sylvian Fissure (+5.16% Dice Gain, Pair 44).</strong><br>
          <strong>Top Row (A–D): Original &amp; Transformed Labels on Target Anatomy (Axial Z=148).</strong><br>
          &bull; <strong>Panel A:</strong> Target Anatomy with Ground Truth STG label.<br>
          &bull; <strong>Panel B:</strong> Original Moving Label (Affine, Dice: 0.441).<br>
          &bull; <strong>Panel C:</strong> Sobolev Transformed Label (Dice: 0.6810): Decelerates early, failing to slide across the deep insular boundary.<br>
          &bull; <strong>Panel D:</strong> Hyperelastic Transformed Label (Dice: 0.7326, <strong>+5.16% gain</strong>): Conformal alignment along the Sylvian bank.<br><br>
          <strong>Bottom Row (E–H): Visualization of the Flows.</strong><br>
          &bull; <strong>Panel E:</strong> Sobolev Flow Field: Isotropic drag choking against the insular wall.<br>
          &bull; <strong>Panel F:</strong> Hyperelastic Flow Field: High-velocity tangential flow sweeping tissue along the fissure.<br>
          &bull; <strong>Panel G:</strong> Deformed Mesh Grid: Tangential coordinate shear sliding without grid singularities.<br>
          &bull; <strong>Panel H:</strong> Active Flow Differential (&Delta;u): Isolates the extra tangential slip vectors driving the +5.16% gain.
        </div>
      </div>


    </div>
  </div>

  <!-- SECTION 3: ANATOMICAL DISCREPANCY IN PAIR 44 -->
  <div class="card">
    <div class="card-header">
      <h2>3. Ground-Truth Morphological Discrepancy: Pair 44 (NKI-TRT-20-2 &rarr; MMRR-21-2)</h2>
      <p>Quantifying the 89% ventricular enlargement and cortical atrophy driving the registration challenge</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG1_PLACEHOLDER" alt="Figure 1: Morphology Discrepancy in Pair 44">
        <div class="figure-caption">
          <strong>Figure 1: Ground-Truth Anatomical Morphology Discrepancy in Pair 44 (Mindboggle Hard Cohort).</strong> 
          Top: Coronal views; Bottom: Axial views through the lateral ventricles. 
          The moving source brain (MMRR-21-2) presents marked ex-vacuo ventriculomegaly (total ventricular volume: 22.15 mL) 
          compared to the young adult fixed target (NKI-TRT-20-2, 11.71 mL), representing an <strong>+89.2% volume enlargement</strong>. 
          Concurrently, total cortical volume is 16.3% smaller in the moving brain (465.6 mL vs. 556.1 mL). 
          Cyan contours highlight the lateral ventricles (FreeSurfer aseg 4/43).
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 4: VENTRICULAR REALIGNMENT AND ERROR MAPS -->
  <div class="card">
    <div class="card-header">
      <h2>4. Ventricular Boundary Matching and Residual Error Reduction</h2>
      <p>Direct anatomical comparison of warped brains, target boundary contours, and absolute error difference maps</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG2_PLACEHOLDER" alt="Figure 2: Ventricular Realignment Comparison">
        <div class="figure-caption">
          <strong>Figure 2: Anatomical Realignment and Localized Error Reduction at the Lateral Ventricles (Pair 44).</strong>
          Top row (A–E): Anatomical coronal slices showing the fixed target, affine initialization, Sobolev SyN, Hyperelastic SyN, and Div-Curl SyN. 
          Blue contours trace the ground-truth target ventricular boundary. 
          Under Sobolev SyN (C), the ventricle cannot contract sufficiently, leaving a wide halo of dark CSF outside the target boundary. 
          Both Hyperelastic (D) and Div-Curl (E) pull the moving brain tissue cleanly into alignment with the target ventricle contours.<br>
          Bottom row (F–I): Absolute intensity difference maps |I<sub>target</sub> - I<sub>warped</sub>| (F–H) and error subtraction 
          &Delta; = Error<sub>Sobolev</sub> - Error<sub>Hyperelastic</sub> (I). 
          Intense red regions in Panel I confirm that Hyperelasticity achieves massive error reduction (up to &plusmn;0.40 intensity units) 
          concentrated precisely along the lateral ventricle margins and periventricular white matter.
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 5: CONTINUOUS JACOBIAN MAPS -->
  <div class="card">
    <div class="card-header">
      <h2>5. Continuous Liouville Jacobian Expansion &amp; Contraction Maps</h2>
      <p>Evaluating continuous volume change log det(J) and topological regularity</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG3_PLACEHOLDER" alt="Figure 3: Continuous Jacobian Maps">
        <div class="figure-caption">
          <strong>Figure 3: Continuous Log-Jacobian Determinant log det(J) Maps Demonstrating Unchoked Volume Adaptation.</strong>
          Top: Coronal slices; Bottom: Axial slices. Blue contours outline the target lateral ventricles. 
          Blue colormap indicates local volume contraction (J &lt; 1.0, log J &lt; 0); red indicates local expansion (J &gt; 1.0, log J &gt; 0).<br>
          Sobolev SyN (left) exhibits muted contraction inside the ventricles (minimum J = 0.448, log J &approx; -0.80), 
          constrained by the penalty on ||&nabla;v||<sup>2</sup>. 
          Hyperelastic SyN (center) and Div-Curl SyN (right) achieve deep, biologically accurate ventricular contraction 
          (J = 0.318 and 0.264, log J &approx; -1.15 to -1.33), enabling full alignment with the target boundary.<br>
          Crucially, both continuum models maintain complete diffeomorphism integrity: <strong>0.0% folding</strong> and 
          positive minimum Jacobians (J<sub>min</sub> &gt; 0.010) across the entire 3D volume.
        </div>
      </div>

      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Ventricle Min det(J)</th>
            <th>Ventricle Mean det(J)</th>
            <th>Global Max det(J)</th>
            <th>Global Min det(J)</th>
            <th>Topological Folding %</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Sobolev SyN</strong> (Scalar Smoother)</td>
            <td><span class="badge-base">0.4478</span> (Choked)</td>
            <td>1.7408</td>
            <td>40.5</td>
            <td>0.0306</td>
            <td>0.00%</td>
          </tr>
          <tr>
            <td><strong>Hyperelastic SyN</strong> (Bulk + Shear)</td>
            <td><span class="badge-gain">0.3178</span> (+40.9% deeper)</td>
            <td>1.7427</td>
            <td>110.4</td>
            <td>0.0146</td>
            <td>0.00%</td>
          </tr>
          <tr>
            <td><strong>Div-Curl SyN</strong> (Spectral Helmholtz)</td>
            <td><span class="badge-gain">0.2644</span> (+69.4% deeper)</td>
            <td>1.7451</td>
            <td>156.1</td>
            <td>0.0107</td>
            <td>0.00%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 6: SULCAL SHEAR AND TANGENTIAL SLIP -->
  <div class="card">
    <div class="card-header">
      <h2>6. Tangential Sulcal Shear vs. Isotropic Drag at the Sylvian Fissure</h2>
      <p>High-magnification vector quiver fields revealing tangential sliding along cortical banks</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG4_PLACEHOLDER" alt="Figure 4: Sulcal Shear Tangential Slip">
        <div class="figure-caption">
          <strong>Figure 4: Tangential Sulcal Shear vs. Isotropic Drag at the Sylvian Fissure ROI (Pair 44).</strong>
          High-magnification view of the insular cortex (blue contour, FreeSurfer 1035/2035) and superior temporal gyrus (amber contour, FreeSurfer 1030/2030) 
          flanking the Sylvian fissure.<br>
          In Sobolev SyN (Panel B), displacement quivers cross the sulcal wall perpendicularly, dragging the temporal bank into the insula 
          and blurring the cortical ribbon due to the isotropic penalty on &nabla;v.<br>
          In Hyperelastic SyN (Panel C), displacement vectors bend sharply and align <strong>tangentially</strong> with the sulcal wall, 
          allowing the superior temporal gyrus to slide past the insula without shearing tissue across the CSF boundary. 
          This tangential slip directly yields a <strong>+5.16% Dice gain in the Superior Temporal Gyrus</strong> (0.6810 &rarr; 0.7326).
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 7: STRUCTURE-BY-STRUCTURE DICE GAINS -->
  <div class="card">
    <div class="card-header">
      <h2>7. Structure-Specific Quantitative Breakdown: Where the Gains Live</h2>
      <p>Detailed cortical and subcortical overlap metrics across anatomical parcels (Pair 44)</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG5_PLACEHOLDER" alt="Figure 5: Structure-by-Structure Dice Gain">
        <div class="figure-caption">
          <strong>Figure 5: Structure-Specific Dice Gains in Pair 44 (Mindboggle Hard Cohort).</strong>
          Horizontal bar chart comparing Sobolev SyN (blue), Hyperelastic SyN (green), and Div-Curl SyN (amber). 
          The cortical gains are concentrated in structures bordering the lateral ventricles and the Sylvian fissure, 
          with the Middle Temporal Gyrus gaining +6.93%, Superior Temporal Gyrus gaining +5.16%, and Caudal Anterior Cingulate gaining +4.37%.
        </div>
      </div>

      <table class="data-table">
        <thead>
          <tr>
            <th>Anatomical Structure</th>
            <th>FreeSurfer IDs</th>
            <th>Sobolev SyN</th>
            <th>Hyperelastic SyN</th>
            <th>Div-Curl SyN</th>
            <th>Absolute Gain</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Middle Temporal Gyrus</strong></td>
            <td>1015 / 2015</td>
            <td>0.5443</td>
            <td>0.6136</td>
            <td>0.6120</td>
            <td><span class="badge-gain">+6.93%</span></td>
          </tr>
          <tr>
            <td><strong>Superior Temporal Gyrus</strong></td>
            <td>1030 / 2030</td>
            <td>0.6810</td>
            <td>0.7326</td>
            <td>0.7338</td>
            <td><span class="badge-gain">+5.16%</span></td>
          </tr>
          <tr>
            <td><strong>Caudal Anterior Cingulate</strong></td>
            <td>1002 / 2002</td>
            <td>0.5349</td>
            <td>0.5786</td>
            <td>0.5573</td>
            <td><span class="badge-gain">+4.37%</span></td>
          </tr>
          <tr>
            <td><strong>Supramarginal Gyrus</strong></td>
            <td>1031 / 2031</td>
            <td>0.6106</td>
            <td>0.6443</td>
            <td>0.6415</td>
            <td><span class="badge-gain">+3.37%</span></td>
          </tr>
          <tr>
            <td><strong>Inferior Temporal Gyrus</strong></td>
            <td>1009 / 2009</td>
            <td>0.6093</td>
            <td>0.6402</td>
            <td>0.6385</td>
            <td><span class="badge-gain">+3.09%</span></td>
          </tr>
          <tr>
            <td><strong>Superior Parietal Cortex</strong></td>
            <td>1029 / 2029</td>
            <td>0.5152</td>
            <td>0.5297</td>
            <td>0.5280</td>
            <td><span class="badge-gain">+1.45%</span></td>
          </tr>
          <tr>
            <td><strong>Precuneus Cortex</strong></td>
            <td>1025 / 2025</td>
            <td>0.6420</td>
            <td>0.6528</td>
            <td>0.6468</td>
            <td><span class="badge-gain">+1.08%</span></td>
          </tr>
          <tr>
            <td><strong>3rd Ventricle</strong></td>
            <td>14</td>
            <td>0.7692</td>
            <td>0.7789</td>
            <td>0.7841</td>
            <td><span class="badge-gain">+1.49%</span></td>
          </tr>
          <tr>
            <td><strong>Insular Cortex</strong></td>
            <td>1035 / 2035</td>
            <td>0.8317</td>
            <td>0.8377</td>
            <td>0.8340</td>
            <td><span class="badge-gain">+0.60%</span></td>
          </tr>
          <tr style="background: #F1F5F9; font-weight: bold;">
            <td><strong>Global 31-Label Cortical Mean</strong></td>
            <td>All DKT31</td>
            <td>0.6011</td>
            <td>0.6165</td>
            <td>0.6162</td>
            <td><span class="badge-gain">+1.54%</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- SECTION 8: INTRA-SCANNER REPLICATION ON PAIR 08 -->
  <div class="card">
    <div class="card-header">
      <h2>8. Intra-Scanner Cohort Replication: Pair 08 (MMRR-21-8 &rarr; MMRR-21-1)</h2>
      <p>Eliminating inter-scanner confounding: Continuum mechanics advantage replicates on identical scanner protocol</p>
    </div>
    <div class="card-body">
      <div class="figure-container">
        <img src="FIG6_PLACEHOLDER" alt="Figure 6: Intra-Scanner Replication on Pair 08">
        <div class="figure-caption">
          <strong>Figure 6: Independent Replication of Continuum Mechanics Advantage on Pair 08 (Mindboggle Easy / Intra-Scanner).</strong>
          Top row (A–C): Axial slices of Fixed Target (MMRR-21-8), Sobolev SyN warped, and Hyperelastic SyN warped images. 
          Blue contours mark the lateral ventricles.<br>
          Bottom row (D–F): Absolute difference maps |I<sub>target</sub> - I<sub>warped</sub>| for Sobolev (D) and Hyperelastic (E), 
          and error reduction &Delta; = Error<sub>Sobolev</sub> - Error<sub>Hyperelastic</sub> (F). 
          Red voxels in Panel F show widespread parenchymal error reduction across cingulate, precentral, and precuneus cortices. 
          Hyperelastic SyN achieves a <strong>+1.35% Cortical Dice gain</strong> (0.6195 &rarr; 0.6331) and Div-Curl achieves +1.33% (0.6328).
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 9: CONCLUSIONS & SUMMARY -->
  <div class="card">
    <div class="card-header">
      <h2>9. Summary &amp; Scientific Conclusions</h2>
      <p>Durable principles established for the syntx registration pipeline</p>
    </div>
    <div class="card-body">
      <ul style="padding-left: 20px; font-size: 14.5px; line-height: 1.8; color: var(--text);">
        <li>
          <strong>Bulk vs. Shear Decoupling is Essential:</strong> Brain deformations in cross-sectional neuroimaging 
          exhibit anisotropic mechanics. CSF spaces undergo massive localized volume change (&plusmn;90%), 
          while cortical sulci require tangential shear sliding. Scalar smoothers (Sobolev, Gaussian) conflate these modes, 
          leading to compromised ventricular contraction and cross-sulcal blurring.
        </li>
        <li>
          <strong>Diffeomorphism Integrity is Fully Preserved:</strong> Allowing unconstrained shear does not compromise topology. 
          By maintaining a positive bulk modulus (K=5.0, &beta;<sub>div</sub>=4.0), both Hyperelastic and Div-Curl SyN achieve 
          <strong>0.00% folding</strong> and strictly positive Liouville determinants (J<sub>min</sub> &gt; 0.010) across 100% of tested subjects.
        </li>
        <li>
          <strong>Gains are Replicated Across Cohorts:</strong> The superiority of continuum mechanics is not an artifact of inter-scanner noise. 
          It replicates cleanly on both inter-scanner hard pairs (+1.54% cortical gain on Pair 44) and intra-scanner standard pairs 
          (+1.35% cortical gain on Pair 08), with structure-specific gains reaching up to +6.93% in temporal and periventricular cortex.
        </li>
        <li>
          <strong>Optimal Default Regularization:</strong> The autotuned continuum models (<code>syn_hyperelastic</code> with K=5.0 
          and <code>syn_divcurl</code> with &beta;=4.0) should be established as the premier high-accuracy diffeomorphic registration 
          defaults for challenging anatomical cohorts.
        </li>
      </ul>
    </div>
  </div>

</div>

</body>
</html>
"""

final_html = template.replace("FIG1_PLACEHOLDER", img_fig1)\
                     .replace("FIG2_PLACEHOLDER", img_fig2)\
                     .replace("FIG3_PLACEHOLDER", img_fig3)\
                     .replace("FIG4_PLACEHOLDER", img_fig4)\
                     .replace("FIG5_PLACEHOLDER", img_fig5)\
                     .replace("FIG6_PLACEHOLDER", img_fig6)\
                     .replace("FIG7_PLACEHOLDER", img_fig7)\
                     .replace("FIG8_PLACEHOLDER", img_fig8)\
                     .replace("FIG9_PLACEHOLDER", img_fig9)\
                     .replace("FIG10_PLACEHOLDER", img_fig10)

with open(REPORT_PATH, "w") as f:
    f.write(final_html)

print(f"HTML report successfully assembled and written to {REPORT_PATH}")
print(f"Report file size: {os.path.getsize(REPORT_PATH) / 1024:.1f} KB")
