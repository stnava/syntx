import json, datetime

bold = json.load(open("/tmp/motion_report/bold_results.json"))
dwi = json.load(open("/tmp/motion_report/dwi_results.json"))
pet = json.load(open("/tmp/motion_report/pet_results.json"))
figs = json.load(open("/tmp/motion_report/figs_partial.json"))
pet_figs = json.load(open("/tmp/motion_report/pet_figs.json"))
new_figs = json.load(open("/tmp/motion_report/new_evidence_figs.json"))
all_figs = {**figs, **pet_figs, **new_figs}

def img(name, caption, width="100%"):
    if name not in all_figs:
        return f'<p style="color:#ef4444">[figure {name} missing]</p>'
    return (f'<figure style="margin:0 0 1.8rem 0">'
            f'<img src="data:image/png;base64,{all_figs[name]}" style="max-width:{width};border:1px solid #27354a;border-radius:8px"/>'
            f'<figcaption style="font-size:0.84rem;color:#94a3b8;margin-top:0.5rem;line-height:1.5">{caption}</figcaption></figure>')

socom_speedup = bold["SOCOM"]["slow_time"] / bold["SOCOM"]["adaptive_time"]
bold_full_speedup = 2019.1 / 40.0
b0_mask_vals = __import__("numpy").array(dwi["bvals"]) <= 50
import numpy as np
dwi_fd = np.array(dwi["old_fd"])
two_mean_fd = np.array(dwi["two_mean_fd"])
b0_fd_old = dwi_fd[b0_mask_vals].mean()
b0_fd_two_mean = two_mean_fd[b0_mask_vals].mean()
b0_fd_trusted = np.mean(dwi["b0_only_fd"])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Motion Correction Across Modalities -- syntx</title>
</head>
<body style="margin:0;padding:2.5rem;background:#0b0f17;color:#e2e8f0;font-family:-apple-system,Segoe UI,sans-serif;line-height:1.6">
<div style="max-width:1200px;margin:0 auto">

<h1 style="font-size:1.8rem;margin-bottom:0.2rem">Motion Correction Across Modalities: Methods, Evidence, and Open Problems</h1>
<div style="color:#64748b;font-size:0.85rem;margin-bottom:2rem">syntx core document &middot; {datetime.datetime.now().strftime('%Y-%m-%d')}</div>

<div style="background:#141b2d;border-left:4px solid #38bdf8;border-radius:6px;padding:1.2rem 1.5rem;margin-bottom:2.5rem;font-size:0.95rem">
<b>What this document is.</b> A practical guide to doing rigid motion correction well on a
4D medical imaging time series, covering three structurally different cases: a single-
contrast series where every frame should look like every other frame (BOLD fMRI, ASL
perfusion), a multi-contrast series where frames come in genuinely different flavors that
must never be blended together (diffusion MRI's b0 + diffusion-weighted volumes), and a
continuously-drifting-contrast series where the image itself is supposed to change over
time for reasons that have nothing to do with head motion (dynamic PET tracer uptake).
Each case needs a different reference-construction strategy and a different idea of what
"fast" can mean -- treating them the same is the single biggest mistake to avoid. This
document shows the method for each, with real data, real numbers, and real before/after
images -- including the one case (PET) that is not yet solved.
</div>

<h2>1. The core problem: what makes a good reference, and what makes a cheap decision correct</h2>
<p>Rigid motion correction registers every frame of a time series to a shared reference
image, recovering a 6-degree-of-freedom (3 translations + 3 rotations) transform per frame.
Two design choices determine whether this works well:</p>
<ol>
<li><b>What is the reference?</b> The obvious choice -- average every frame together -- is
only correct when every frame genuinely has the same contrast and real motion is the only
thing varying. It breaks in two different ways: (a) if the series mixes multiple real
contrasts (DWI), the average is dominated by whichever contrast has more frames, and
registering the minority contrast against it is like registering two different scan types
-- the registration can "succeed" at a wrong answer, reporting apparent motion that is
really a contrast mismatch; (b) if real motion is non-trivial, the average itself is a
motion-blurred image, which degrades every subsequent registration against it.</li>
<li><b>How does a caller cheaply decide "this frame doesn't need expensive registration"?</b>
The two intuitive shortcuts -- raw image correlation, and center-of-mass displacement --
both fail in ways that are invisible until tested on real data at real scale (&sect;2).
The one that works measures the actual quantity of interest (a real rigid transform's own
FD) cheaply, instead of a proxy for it.</li>
</ol>

<h2>2. Picking a cheap "does this frame need more work" signal</h2>
<p>A production pipeline needs a fast pre-check: most frames in most series need little or
no correction, and paying full registration cost for every frame is wasted work. Three
candidates, tested on real data:</p>
<table style="width:100%;border-collapse:collapse;margin:1.2rem 0">
<thead><tr><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Candidate signal</th><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Why it seems reasonable</th><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Why it fails</th></tr></thead>
<tbody>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Raw voxelwise correlation to reference</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Cheap, no registration needed</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Saturates near 1.0 on smooth, low-resolution images regardless of real motion -- on a real 3.5mm-resolution series this silently skipped registering <i>every single frame</i>, reporting zero motion when the true motion was 0.28mm mean.</td></tr>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Center-of-mass displacement</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Resolution-independent (physical mm), cheap (one image moment calculation)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Blind to rotation (a pure rotation about the head's own center does not move its center of mass). On real BOLD data, rotation contributes roughly as much as translation to true FD (~1:1 ratio measured) -- this signal misses about half the real motion.</td></tr>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b"><b>A cheap, coarse, real registration's own recovered FD</b></td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Not a proxy -- the actual quantity, computed at low cost (downsampled images, narrow capture range, short optimization schedule)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b"><b>This is the one that works.</b> Resolution- and modality-independent by construction, since it is literal physical-space displacement, not an intensity statistic.</td></tr>
</tbody>
</table>
<p>This is implemented as <code>syntx.batched_rigid_register_pass_adaptive</code> /
<code>backend='pytorch_batched_adaptive'</code>: every frame gets one cheap coarse
registration; frames whose coarse FD exceeds a threshold get a second, finer pass; frames
below it keep the cheap result. The gate signal and the final answer are the same kind of
measurement at two resolutions, not two different things.</p>

<h2>3. BOLD fMRI / ASL perfusion -- single-contrast series, the straightforward case</h2>
<p>Every frame in a resting-state BOLD or ASL perfusion series is meant to look like every
other frame; real head motion is the only source of frame-to-frame difference that
matters. This is the case the adaptive gate above is built for directly.</p>

<h3>3.1 Evidence: a real, large motion event, before and after</h3>
<p>The clearest way to confirm a motion-correction method actually works is to find a real,
large displacement in real data and look at it directly, not just read a summary
statistic. In a real 25-volume ASL perfusion series, one frame shows 6.1mm of real head
displacement (more than 3 voxels at this acquisition's resolution) -- large enough to see
with the naked eye in a checkerboard comparison against the reference:</p>
{img("asl_checkerboard_zoom", "Zoomed checkerboard tiling of a real frame against the reference image, before and after motion correction, at the single largest real motion event found in this ASL series (6.1mm FD). Before correction, tissue boundaries break across tile seams; after correction, they run continuously.")}
<p>The same check on a full real 750-volume resting-state BOLD series finds its largest
event at 2.2mm FD:</p>
{img("bold_checkerboard", "Checkerboard comparison at the largest real motion event in a full 750-volume resting-state BOLD series (frame 671, FD=2.20mm).")}
{img("bold_full_fd_trace", "The complete real FD trace for this 750-volume series (computed in 43 seconds), with the event shown above marked.")}

<h3>3.2 Evidence: speed, at the scale that actually matters</h3>
<p>The reason this needs to be fast at all: a 750-volume real series is not a synthetic
stress test, it is an ordinary resting-state fMRI acquisition. A per-frame registration
approach (correct, but one independent optimization per frame) takes real, substantial
time at this scale:</p>
{img("bold_timing", "Wall-clock time (log scale) for a 64-frame truncation of two real subjects, comparing a per-frame baseline against the adaptive gate.")}
<p><b>At full scale (not truncated):</b> the complete 750-volume series runs in <b>40.0
seconds</b> against a <b>2019-second (~34 minute)</b> per-frame baseline -- a <b>{bold_full_speedup:.0f}x</b>
speed-up, with no crash and a plausible resulting motion estimate (mean FD 0.062mm, 5.2%
of frames above the standard 0.5mm high-motion threshold -- a normal profile for a
cooperative subject).</p>

<h2>4. Diffusion MRI (DWI) -- multi-contrast series, the harder case</h2>
<p>A DWI acquisition is not one contrast repeated -- it interleaves b0 volumes (no
diffusion weighting, bright, T2-like contrast) with diffusion-weighted volumes at one or
more b-values (directionally attenuated, often substantially darker in white matter along
the gradient direction). These are genuinely different images of the same anatomy, not
noisy repeats of the same image. Treating them as one undifferentiated series -- averaging
all of them into one reference -- creates a reference that matches neither contrast well,
and registering the numerically smaller group (often the b0 volumes) against it is
registering across a contrast mismatch, not correcting for motion.</p>

<h3>4.1 The method: one mean per contrast, one careful registration between them</h3>
<p>The correct structure keeps every per-frame registration same-contrast (where the fast
gate from &sect;2 is valid) and isolates the one unavoidable cross-contrast problem into a
single, affordable step:</p>
<ol>
<li>Build a sharp mean for the b0 group, and a separate sharp mean for the diffusion-weighted
group, independently (each a same-contrast, small-motion problem -- the coarse-FD-ranked
low-motion-subset builder from &sect;1 handles this cheaply).</li>
<li>Register the two means to each other <b>exactly once</b>, with a wide capture-range
search (this is the only cross-contrast registration anywhere in the pipeline, so it can
afford to be careful rather than fast).</li>
<li>Motion-correct every b0 frame to the b0 mean, and every diffusion-weighted frame to the
diffusion-weighted mean -- both same-contrast, small-motion problems, both handled by the
fast adaptive gate from &sect;2.</li>
<li>Compose the diffusion-weighted group's correction with the one mean-to-mean transform,
landing every frame in one shared physical space.</li>
</ol>

<h3>4.2 Evidence: the cross-contrast registration itself</h3>
{img("dwi_cross_reg_edge_overlay", "Edge-overlay comparison (not a checkerboard -- checkerboarding two genuinely different contrasts confounds 'this tile looks different' with 'this tile is misaligned', which is exactly the failure mode this whole section exists to avoid). The B0 mean's own anatomical boundaries (green) are traced once and overlaid, unchanged, on the DWI mean before and after the one cross-contrast registration: before, the boundary sits slightly off the DWI mean's own ventricle/sulcal edges; after, it tracks them precisely. The two means were already reasonably close in this real subject (no gross head repositioning occurred between the b0 and diffusion-weighted portions of one continuous scan) -- the point of this step is not correcting a large offset, it is establishing a reliable, one-time anatomical correspondence that every per-frame registration can then rely on, instead of each of 36 diffusion-weighted frames independently struggling with the same cross-contrast problem.")}

<h3>4.3 Evidence: why this matters -- three attempts, real numbers</h3>
<p>Real SOCOM diffusion data (4 b0 + 36 diffusion-weighted volumes, two b-values) with an
independent, trusted ground truth available: registering the b0 volumes only among
themselves (impossible for the diffusion-weighted volumes, which have no same-contrast
partner to check against, but a valid internal check for b0 alone) gives 0.748mm mean FD.
Three different attempts at correcting the whole series were compared against that
ground truth:</p>
{img("dwi_b0_fd_comparison", "b0 apparent motion under four conditions: the original blended-mean approach, an intermediate attempt (one shared b0-only reference, but still registering every frame -- including diffusion-weighted ones -- directly to it), the two-mean design from &sect;4.1, and the trusted b0-only-internal ground truth.")}
<p>The intermediate attempt -- building a clean b0 reference but still registering every
diffusion-weighted frame <i>directly</i> to it with the same fast, narrow-capture-range
gate used for same-contrast BOLD data -- made things <i>worse</i> than the original method
(7.28mm vs. 4.66mm mean b0 FD), not better. The narrow capture range that is exactly right
for same-contrast, small-motion data actively hurts cross-contrast registration, which
needs a wide search even though the real head motion involved is small. The two-mean
design resolves this by never asking the fast narrow-search gate to solve a cross-contrast
problem at all: it reaches <b>0.796mm</b>, matching the trusted ground truth (0.748mm)
closely -- down from the original method's 4.66mm.</p>

<h2>5. Dynamic PET -- continuously-drifting contrast, an open problem</h2>
<p>Dynamic PET is not a harder version of the DWI problem, it is a different problem.
DWI's contrast varies across a small number of discrete, repeatable categories (one b0
contrast, one or two diffusion-weighted contrasts) -- each category has multiple frames
that genuinely share a contrast, so a same-contrast mean is a meaningful, well-defined
target. Dynamic PET's contrast changes continuously and <b>monotonically</b> with tracer
uptake: no two frames in the whole series necessarily share a contrast, and the earliest
frames can be close to blank.</p>

<h3>5.1 Evidence: the problem has nothing to do with registration</h3>
<p>The single clearest way to see why standard motion correction does not transfer to this
case: look at the same anatomical slice at different real time points in a real dynamic
FDG-PET acquisition.</p>
{img("pet_uptake_drift_evidence", "The same axial slice of the same real subject at 5 time points across a real dynamic FDG-PET acquisition. No frame-to-frame head motion is implied or assumed here -- the dramatic visual change from an essentially blank image to a fully tracer-filled brain is tracer kinetics alone. Any method that registers each frame to a single fixed reference image is, by construction, trying to explain this uptake curve as if it were spatial displacement.")}

<h3>5.2 Evidence: a "high motion" frame that isn't really about motion</h3>
<p>Running the same adaptive-gate machinery used successfully for BOLD/ASL on a real
dynamic FDG-PET series does produce large recovered FD values -- but checking what the
registration is actually doing shows it is not finding a spatial shift the way it does for
BOLD or DWI:</p>
{img("pet_edge_overlay", "An edge overlay, not a checkerboard, for the same reason as &sect;4.2: the reference's own anatomical boundary (green) is traced once and overlaid unchanged on the real frame with the largest recovered FD (9.40mm) in a 32-frame post-uptake window, before and after the registration 'correction'. The contour already sits exactly on the brain boundary in the RAW frame -- it does not move after 'correction', because there was never a real spatial offset to fix. The high FD number is an artifact of the registration optimizer finding some transform that improves the Mattes mutual-information score against a reference with different uptake, not evidence of head motion.")}

<h3>5.3 Status: not solved by reference-strategy changes -- but feature-based registration looks promising</h3>
<p>Two natural next ideas were tried on real data and neither is a real fix:</p>
<ul>
<li><b>Treat it like DWI (anchor to a low-variability subset):</b> does not transfer --
DWI's fix works because discrete contrast GROUPS exist with genuine same-contrast
repeats; dynamic PET's drift is continuous, so "the frames most similar to the crude mean"
is not the same concept as "the frames sharing group A's true contrast." Tested directly:
FD was not meaningfully better than the naive whole-series mean (and at points slightly
worse), and even the per-frame trusted baseline itself shows elevated FD (around 3mm
mean) in a stable post-uptake window -- the elevated numbers are not an artifact of the
fast method, they reflect that intensity-based rigid registration is answering the wrong
question for this data.</li>
<li><b>Restrict to a late, stable-uptake window:</b> helps avoid the earliest near-blank
frames (confirmed: the first ~8 frames of this real series are literally zero intensity,
before tracer arrival -- any registration against them is meaningless by construction),
but does not resolve the continuous-drift problem within the stable window itself.</li>
</ul>

<h4>A third idea that changes the actual registration mechanism, not just the reference: feature-based (SIFT3D) rigid estimation</h4>
<p>Every attempt so far kept the same core mechanism -- optimize a rigid transform against
voxel-intensity mutual information -- and only changed which image is used as the target.
A structurally different approach is to stop comparing raw intensities at all: detect
distinctive 3D keypoints (<code>syntx.landmarks.detect_sift3d</code>, a full 3D SIFT:
difference-of-Gaussian keypoint detection plus physical-space gradient-histogram
descriptors), match them between reference and frame by descriptor similarity, and fit a
rigid transform to the matched point pairs with RANSAC. A gradient-histogram descriptor
(especially with local contrast normalization) describes the <i>shape</i> of local
structure, not its absolute brightness -- in principle much less sensitive to a global
uptake-driven intensity change than direct voxel-intensity comparison.</p>
<p>Tested directly on the same real series, at native resolution (the earlier intensity-based
numbers used a 4mm-downsampled series for speed; SIFT3D needs the real spatial detail that
downsampling throws away):</p>
{img("pet_sift3d_comparison", "SIFT3D feature-based rigid displacement estimate vs. the intensity-based (Mattes-MI) FD, for the same 11 real frames, at native PET resolution. Mattes-MI swings erratically across a 0-9.4mm range with no physiologically plausible pattern. SIFT3D, wherever it finds enough matched keypoints to fit a rigid transform at all, gives a narrow, stable 0.6-2.1mm range -- and fails cleanly (no fit, rather than a wrong answer) on the earliest, lowest-structure frames, which is the physically correct behavior.")}
<p>This is a genuinely promising, but not yet validated, direction: matched keypoint counts
are sparse (4-14 points per frame at a 512-keypoint detection budget), there is no
independent ground-truth motion trace for this public dataset to confirm the SIFT3D numbers
are the <i>correct</i> ones rather than merely more stable-looking, and the earliest frames
still cannot be registered by any method because they contain almost no real structure to
match. It is reported here as the most promising lead found, not as a solved problem.</p>
<p>What dynamic PET's remaining gap actually needs, beyond this lead, is outside the scope of
rigid-registration-to-a-fixed-reference: either a kinetics-aware reference (e.g. fitting and subtracting an expected
uptake curve before comparing frames), grouping frames by acquisition-protocol-defined
time bins rather than image similarity, or accepting that per-frame-precision motion
correction is not worth its cost for a given downstream use case (as this ecosystem's own
template-building pipeline already concluded independently, settling for no motion
correction at all rather than a method that would silently produce a confident-looking
but wrong answer). This document reports that conclusion rather than disguising it as
solved.</p>

<h2>6. What this means in practice</h2>
<table style="width:100%;border-collapse:collapse;margin:1.2rem 0">
<thead><tr><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Series type</th><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Reference strategy</th><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Per-frame search width</th><th style="text-align:left;padding:0.5rem;color:#94a3b8;border-bottom:1px solid #27354a">Status</th></tr></thead>
<tbody>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b">BOLD / ASL (single contrast)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Low-motion-subset mean</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Narrow (fast)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Solved, validated at full real scale (~50x speed-up)</td></tr>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b">DWI / SlowFlow (discrete multi-contrast)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">One mean per contrast group</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Narrow within group; wide for the one cross-group registration</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Solved, validated on real data (matches trusted ground truth)</td></tr>
<tr><td style="padding:0.5rem;border-bottom:1px solid #1e293b">Dynamic PET (continuous contrast drift)</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">No reference-strategy fix found</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b">N/A -- mechanism itself (intensity-based) is the wrong tool</td><td style="padding:0.5rem;border-bottom:1px solid #1e293b"><b>Not solved</b>; a feature-based (SIFT3D) rigid estimate is a promising, not-yet-validated lead (&sect;5.3)</td></tr>
</tbody></table>

<h2>7. What shipped in syntx</h2>
<ul>
<li><code>syntx.batched_rigid_register_pass_adaptive</code> / <code>backend='pytorch_batched_adaptive'</code>
-- registration-derived motion gate (&sect;2).</li>
<li><code>syntx.build_low_motion_reference</code> -- sharp, same-contrast reference from a
cheaply-ranked low-motion frame subset.</li>
<li><code>syntx.motion_correct_grouped</code> -- the two-mean, one-cross-registration design
for multi-contrast series (&sect;4.1).</li>
<li>48 tests across <code>test_motion_batched.py</code> and <code>test_motion_reference.py</code>,
all passing; full existing motion test suite re-verified with no regressions.</li>
</ul>

<footer style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #27354a;color:#64748b;font-size:0.8rem">
syntx core document &middot; reproduction scripts in <code>scripts/motion_correction_report/</code>
</footer>
</div>
</body>
</html>
"""

open("/tmp/motion_report/report_v2.html", "w").write(html)
print("wrote report_v2.html,", len(html), "bytes")
