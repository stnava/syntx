#!/usr/bin/env python3
"""
3D deep parameter sweep for TVF and SyNGS on Pair 44 (mbhard).
- One MPS job at a time (strict rule)
- Fixed: cc2 similarity, reg_iterations=[100,100,20]
- Hard folding guard: skip deployment if fold > FOLD_LIMIT
- Results appended to JSON after every run (crash-safe)

Run:  python scripts/sweep_3d_tvf_syngs.py
      python scripts/sweep_3d_tvf_syngs.py --method tvf
      python scripts/sweep_3d_tvf_syngs.py --method syngs
      python scripts/sweep_3d_tvf_syngs.py --resume   (skip already-done configs)
"""
import argparse, json, os, sys, time
from syntx.benchmark.evaluate import evaluate_mindboggle_pair
from syntx.benchmark.config import get_model_config

RESULTS_FILE = "results/sweep_3d_results.json"
PAIR_IDX = 44
DEVICE = "mps"
FOLD_LIMIT = 0.005   # hard 3D folding threshold (%)

# Baselines (post normalization-fix, pre this sweep)
TVF_BASELINE  = {"dice": 0.61202, "fold": 0.00194, "time": 264.9}
SYNGS_BASELINE = {"dice": 0.60212, "fold": 0.02181, "time": 123.0}  # post-8406f7c deterministic antithetic; old 0.61602 was non-deterministic pre-8406f7c code

os.makedirs("results", exist_ok=True)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_result(key, rec):
    results = load_results()
    results[key] = rec
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved] {key}")


def run_one(method, label, **kwargs):
    """Run evaluate_mindboggle_pair, print and return result dict."""
    key = f"{method}_{label}"
    existing = load_results()
    if key in existing and args.resume:
        r = existing[key]
        status = "✓ fold_ok" if r["fold"] <= FOLD_LIMIT else "✗ fold_violated"
        print(f"  [skip] {label:<55s} Dice={r['dice']:.5f}  fold={r['fold']:.5f}%  {status}")
        return r

    t0 = time.time()
    try:
        res = evaluate_mindboggle_pair(
            pair_idx=PAIR_IDX, model=method, device=DEVICE, verbose=False, **kwargs
        )
        elapsed = time.time() - t0
        rec = {
            "method": method,
            "label": label,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "dice": float(res["dice_sym"]),
            "dice_fixed": float(res["dice_fixed"]),
            "dice_moving": float(res["dice_moving"]),
            "fold": float(res["folding_pct"]),
            "min_jac": float(res["min_jacobian"]),
            "time": elapsed,
            "fold_ok": res["folding_pct"] <= FOLD_LIMIT,
        }
    except Exception as e:
        elapsed = time.time() - t0
        rec = {"method": method, "label": label, "kwargs": {k: str(v) for k, v in kwargs.items()},
               "error": str(e), "dice": 0.0, "fold": 99.0, "fold_ok": False, "time": elapsed}

    baseline_dice = TVF_BASELINE["dice"] if method == "tvf" else SYNGS_BASELINE["dice"]
    delta = rec["dice"] - baseline_dice
    fold_flag = "✓" if rec.get("fold_ok") else "✗ FOLD"
    print(f"  {label:<55s} Dice={rec['dice']:.5f}  Δ={delta:+.5f}  fold={rec['fold']:.5f}%  {fold_flag}  t={rec['time']:.0f}s")

    save_result(key, rec)
    return rec


# ──────────────────────────────────────────────────────────────────────────────
def sweep_tvf():
    print("\n" + "="*70)
    print("TVF 3D SWEEPS  (baseline Dice=0.61202, fold=0.00194%)")
    print("="*70)

    cfg = get_model_config("tvf")

    FIXED = dict(
        optimizer=cfg["optimizer"], optimizer_lr=cfg["optimizer_lr"],
        flow_sigma=3.0,  # spectral gate: any positive value = on; value itself is inert for dsti1
        total_sigma=cfg["tvf_total_sigma"],
        regularizer=cfg["tvf_regularizer"], dsti_alpha=cfg["dsti_alpha"],
        max_step_norm=cfg["max_step_norm"],
        n_time_steps=cfg["tvf_n_time_steps"],
        cfl_momentum=cfg["tvf_cfl_momentum"],
    )
    # Drop keys from FIXED that don't exist in current config (multipoint_loss, constant_speed, sobolev_alpha removed)


    # ── Sweep A: gaussian + total_sigma (can elastic regularization prevent folding?) ──
    print("\n── Sweep A: gaussian regularizer + elastic total_sigma ──")
    for ts in [0.05, 0.1, 0.2, 0.5, 1.0]:
        kw = dict(FIXED); kw.update(regularizer="gaussian", total_sigma=ts)
        run_one("tvf", f"gaussian_ts{ts}", **kw)

    # ── Sweep A2: gaussian + smaller max_step_norm ────────────────────────────
    print("\n── Sweep A2: gaussian + reduced max_step_norm ──")
    for msn in [0.15, 0.20, 0.25, 0.35]:
        kw = dict(FIXED); kw.update(regularizer="gaussian", max_step_norm=msn)
        run_one("tvf", f"gaussian_msn{msn}", **kw)

    # ── Sweep B: dsti1 + flow_sigma (3D-native, evaluate.py bug now fixed) ────
    print("\n── Sweep B: dsti1 + flow_sigma ──")
    for fs in [0.5, 1.5, 2.0, 2.5, 3.0]:
        kw = dict(FIXED); kw["flow_sigma"] = fs
        run_one("tvf", f"dsti1_fs{fs}", **kw)

    # ── Sweep C: dsti1 + max_step_norm ────────────────────────────────────────
    print("\n── Sweep C: dsti1 + max_step_norm ──")
    for msn in [0.25, 0.35, 0.60, 0.70]:
        kw = dict(FIXED); kw["max_step_norm"] = msn
        run_one("tvf", f"dsti1_msn{msn}", **kw)

    # ── Sweep D: dsti1 + optimizer_lr ─────────────────────────────────────────
    print("\n── Sweep D: dsti1 + optimizer_lr ──")
    for lr in [0.6, 0.8, 1.0, 1.5, 2.0]:
        kw = dict(FIXED); kw["optimizer_lr"] = lr
        run_one("tvf", f"dsti1_lr{lr}", **kw)

    # ── Sweep E: dsti1 + total_sigma (elastic regularization alone) ───────────
    print("\n── Sweep E: dsti1 + total_sigma ──")
    for ts in [0.0, 0.05, 0.1, 0.2, 0.5]:
        kw = dict(FIXED); kw["total_sigma"] = ts
        run_one("tvf", f"dsti1_ts{ts}", **kw)

    # ── Sweep F: dsti1 + multipoint_loss ──────────────────────────────────────
    print("\n── Sweep F: dsti1 + multipoint_loss ──")
    for mp in [[0.5], [0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]]:
        label = "mp_" + "_".join(str(v) for v in mp)
        kw = dict(FIXED); kw["multipoint_loss"] = mp
        run_one("tvf", label, **kw)

    # ── Sweep G: best promising combination (winner from above) ───────────────
    print("\n── Sweep G: combined best candidates (2-factor) ──")
    results = load_results()
    # Find best fold-clean dsti1 result from sweeps B-F
    candidates = [
        (k, v) for k, v in results.items()
        if k.startswith("tvf_dsti1") and v.get("fold_ok") and v["dice"] > TVF_BASELINE["dice"]
    ]
    if candidates:
        best_k, best_v = max(candidates, key=lambda x: x[1]["dice"])
        print(f"  Best fold-clean dsti1 so far: {best_k}, Dice={best_v['dice']:.5f}")
        best_kwargs = {k: v for k, v in best_v["kwargs"].items() if k not in ("regularizer",)}
        # Try combining best dsti1 with increased flow_sigma
        try:
            best_fs = float(best_v["kwargs"].get("flow_sigma", cfg["tvf_flow_sigma"]))
            best_msn = float(best_v["kwargs"].get("max_step_norm", cfg["max_step_norm"]))
            if best_fs != 2.0:
                kw = dict(FIXED); kw.update(flow_sigma=best_fs, max_step_norm=best_msn)
                run_one("tvf", f"combo_fs{best_fs}_msn{best_msn}", **kw)
        except Exception:
            pass

    # ── Sweep H: sobolev regularizer (different from both dsti1 and gaussian) ─
    print("\n── Sweep H: sobolev regularizer ──")
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        kw = dict(FIXED); kw.update(regularizer="sobolev", sobolev_alpha=alpha)
        run_one("tvf", f"sobolev_alpha{alpha}", **kw)

    # ── Sweep I: n_time_steps ─────────────────────────────────────────────────
    print("\n── Sweep I: n_time_steps ──")
    for nts in [1, 2, 5, 7]:
        kw = dict(FIXED); kw["n_time_steps"] = nts
        run_one("tvf", f"nts{nts}", **kw)

    # ── Sweep J: dsti1 + alpha (KEY KNOB — flow_sigma does nothing for dsti1) ──
    # Confirmed: flow_sigma sweep gave identical Dice=0.61202 for fs in [0.5..3.0]
    # alpha is the actual DST-I regularization strength parameter.
    print("\n── Sweep J: dsti1 + alpha (actual regularization strength) ──")
    for alpha in [0.005, 0.010, 0.015, 0.020, 0.050, 0.10, 0.20, 0.50]:
        kw = dict(FIXED)
        kw["sobolev_alpha"] = alpha   # popped first by evaluate.py TVF block
        kw["dsti_alpha"] = alpha      # also set for completeness
        run_one("tvf", f"dsti1_alpha{alpha}", **kw)

    # ── Sweep K: dsti1 + alpha + max_step_norm combos (if alpha sweep finds winner) ──
    print("\n── Sweep K: dsti1 best_alpha + max_step_norm (msn >= 0.5) ──")
    # Test msn > 0.5 with DEFAULT alpha first (msn < 0.5 already confirmed as worse)
    all_r = load_results()
    alpha_winners = [(k, v) for k, v in all_r.items()
                     if k.startswith("tvf_dsti1_alpha") and v.get("fold_ok") and v["dice"] > TVF_BASELINE["dice"]]
    best_alpha_val = 0.035  # default fallback
    if alpha_winners:
        best_alpha_k, best_alpha_v = max(alpha_winners, key=lambda x: x[1]["dice"])
        best_alpha_val = float(best_alpha_v["kwargs"].get("sobolev_alpha", 0.035))
        print(f"  Best alpha: {best_alpha_val} (Dice={best_alpha_v['dice']:.5f})")
    for msn in [0.60, 0.70, 0.80]:
        kw = dict(FIXED)
        kw["sobolev_alpha"] = best_alpha_val
        kw["dsti_alpha"] = best_alpha_val
        kw["max_step_norm"] = msn
        run_one("tvf", f"dsti1_alpha{best_alpha_val}_msn{msn}", **kw)

    # ── Sweep L: dsti1 + total_sigma FINE (0 → 0.05, searching fold-clean zone) ──
    # KEY FINDING: ts=0.0 → Dice=0.62346 (+0.011), fold=0.797%
    #              ts=0.05 → Dice=0.61202 (locked to baseline), fold=0.00194%
    # Searching for fold-clean improvement between these endpoints.
    print("\n── Sweep L: dsti1 + total_sigma fine (0.001 → 0.045) ──")
    for ts in [0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.045]:
        kw = dict(FIXED)
        kw["total_sigma"] = ts
        run_one("tvf", f"dsti1_ts_fine{ts}", **kw)

    # ── Sweep M: dsti1 + max_step_norm fine (0.50 → 0.60, finding fold boundary) ──
    # KEY FINDING: msn=0.5 → Dice=0.61202, fold=0.00194%
    #              msn=0.6 → Dice=0.62009 (+0.008), fold=0.01137% (2× limit)
    # Searching for largest fold-clean max_step_norm.
    print("\n── Sweep M: dsti1 + max_step_norm fine (0.51 → 0.59) ──")
    for msn in [0.51, 0.52, 0.53, 0.54, 0.55, 0.57]:
        kw = dict(FIXED)
        kw["max_step_norm"] = msn
        run_one("tvf", f"dsti1_msn_fine{msn}", **kw)

    # ── Sweep N: best alpha + total_sigma=0.0 combos ──────────────────────────
    # ts=0.0 gives +0.011 Dice — try with lower alpha (less aggressive) to cut folding
    print("\n── Sweep N: dsti1 + ts=0.0 + alpha sweep (can lower alpha reduce folding?) ──")
    for alpha in [0.05, 0.10, 0.20]:
        kw = dict(FIXED)
        kw["total_sigma"] = 0.0
        kw["sobolev_alpha"] = alpha
        kw["dsti_alpha"] = alpha
        run_one("tvf", f"dsti1_ts0_alpha{alpha}", **kw)

    # ── Sweep P: sobolev regularizer (NOW FIXED — boundary tapering applied) ───
    # Root cause: sobolev path in TVFModel used raw g_process instead of g_process_tapered,
    # causing Gibbs ringing via FFT periodic BCs. Fix applied 2026-09-19 (commit pending).
    # Also: alpha scale for sobolev must match dsti1 range (0.005–0.05), not 0.5–2.0.
    # Previous sweep (alpha=0.5–2.0) was completely over-smoothed → Dice=0.37–0.43.
    print("\n── Sweep P: sobolev regularizer (FIXED) + correct alpha range ──")
    FIXED_SOB = dict(FIXED)
    FIXED_SOB["regularizer"] = "sobolev"
    for al in [0.005, 0.010, 0.020, 0.035, 0.050, 0.075, 0.10]:
        kw = dict(FIXED_SOB); kw["dsti_alpha"] = al
        run_one("tvf", f"sobolev_fixed_alpha{al}", **kw)

    # ── Sweep Q: sobolev + nts=2 (best nts from Sweep I) ────────────────────────
    print("\n── Sweep Q: sobolev + nts=2 combo ──")
    for al in [0.010, 0.020, 0.035]:
        kw = dict(FIXED_SOB); kw["dsti_alpha"] = al; kw["n_time_steps"] = 2
        run_one("tvf", f"sobolev_nts2_alpha{al}", **kw)
    # ── Sweep R: alpha at nts=5 (more fold headroom → can lower alpha further) ──────────
    # nts=5 at baseline alpha=0.035 gives fold=0.00148% (~3.4× headroom vs limit of 0.005%).
    # Strategy: fix nts=5, sweep alpha below 0.035. Less regularization → higher Dice.
    # If fold-clean alpha < 0.035 exists at nts=5, that wins over Sweep J (nts=3).
    print("\n── Sweep R: dsti1 + nts=5 + alpha sweep (exploit fold headroom) ──")
    for al in [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030]:
        kw = dict(FIXED); kw["n_time_steps"] = 5; kw["dsti_alpha"] = al
        run_one("tvf", f"dsti1_nts5_alpha{al}", **kw)

    # ── Sweep S: alpha at nts=3 fine (0.015–0.030, between J values) ──────────
    # Fill gaps in Sweep J to find exact fold-clean alpha at nts=3.
    print("\n── Sweep S: dsti1 + nts=3 + fine alpha (fill J gaps) ──")
    for al in [0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030]:
        kw = dict(FIXED); kw["dsti_alpha"] = al
        run_one("tvf", f"dsti1_nts3_alpha{al}", **kw)


    print("\n" + "="*70)
    print("TVF SWEEP SUMMARY")
    all_results = load_results()
    tvf_results = [(k, v) for k, v in all_results.items()
                   if k.startswith("tvf_") and "error" not in v and v.get("fold_ok")]
    tvf_results.sort(key=lambda x: x[1]["dice"], reverse=True)
    print(f"  {'Config':<50s} {'Dice':>8s} {'Δ':>8s} {'Fold%':>8s}")
    for k, v in tvf_results[:10]:
        delta = v["dice"] - TVF_BASELINE["dice"]
        print(f"  {k:<50s} {v['dice']:>8.5f} {delta:>+8.5f} {v['fold']:>8.5f}%")


# ──────────────────────────────────────────────────────────────────────────────
def sweep_syngs():
    print(f"\n{'='*70}")
    print(f"SyNGS 3D SWEEPS  (baseline Dice={SYNGS_BASELINE['dice']:.5f}, fold={SYNGS_BASELINE['fold']:.5f}%, FOLDS)")
    print(f"{'='*70}")


    cfg = get_model_config("syngs")

    FIXED = dict(
        optimizer=cfg["optimizer"], optimizer_lr=cfg["optimizer_lr"],
        flow_sigma=cfg["flow_sigma"], total_sigma=cfg["total_sigma"],
        alpha=cfg["alpha"], regularizer=cfg["regularizer"],
        max_step_norm=cfg["max_step_norm"], n_steps=cfg["n_steps"],
        # bootstrap_mode NOT set here: evaluate.py defaults to 'antithetic' which matches baseline
    )

    # ── Sweep A: alpha (Sobolev strength) ─────────────────────────────────────
    print("\n── Sweep A: Sobolev alpha ──")
    for alpha in [0.1, 0.2, 0.5, 0.7, 1.0, 1.5]:
        kw = dict(FIXED); kw["alpha"] = alpha
        run_one("syngs", f"alpha{alpha}", **kw)

    # ── Sweep B: flow_sigma ───────────────────────────────────────────────────
    print("\n── Sweep B: flow_sigma ──")
    for fs in [1.0, 2.0, 4.0, 5.0, 6.0]:
        kw = dict(FIXED); kw["flow_sigma"] = fs
        run_one("syngs", f"fs{fs}", **kw)

    # ── Sweep C: max_step_norm (small increments around safe 0.20) ────────────
    print("\n── Sweep C: max_step_norm ──")
    for msn in [0.10, 0.15, 0.25, 0.30, 0.40]:
        kw = dict(FIXED); kw["max_step_norm"] = msn
        run_one("syngs", f"msn{msn}", **kw)

    # ── Sweep D: optimizer_lr ─────────────────────────────────────────────────
    print("\n── Sweep D: optimizer_lr ──")
    for lr in [0.5, 0.7, 1.5, 2.0]:
        kw = dict(FIXED); kw["optimizer_lr"] = lr
        run_one("syngs", f"lr{lr}", **kw)

    # ── Sweep E: n_steps ──────────────────────────────────────────────────────
    print("\n── Sweep E: n_steps (EPDiff integration) ──")
    for ns in [4, 6, 12, 16]:
        kw = dict(FIXED); kw["n_steps"] = ns
        run_one("syngs", f"ns{ns}", **kw)

    # ── Sweep F: total_sigma (elastic regularization) ─────────────────────────
    print("\n── Sweep F: total_sigma ──")
    for ts in [0.05, 0.1, 0.2, 0.5]:
        kw = dict(FIXED); kw["total_sigma"] = ts
        run_one("syngs", f"ts{ts}", **kw)

    # ── Sweep G: bootstrap_mode ───────────────────────────────────────────────
    print("\n── Sweep G: bootstrap_mode ──")
    for bm in ["none", "jitter"]:  # antithetic is the baseline default
        kw = dict(FIXED); kw["bootstrap_mode"] = bm
        run_one("syngs", f"boot_{bm}", **kw)

    # ── Sweep H: combined best ────────────────────────────────────────────────
    print("\n── Sweep H: combined best candidates ──")
    all_results = load_results()
    candidates = [
        (k, v) for k, v in all_results.items()
        if k.startswith("syngs_") and v.get("fold_ok") and v["dice"] > SYNGS_BASELINE["dice"]
    ]
    if candidates:
        best_k, best_v = max(candidates, key=lambda x: x[1]["dice"])
        print(f"  Best fold-clean SyNGS so far: {best_k}, Dice={best_v['dice']:.5f}")
        kw = dict(FIXED)
        try:
            best_alpha = float(best_v["kwargs"].get("alpha", cfg["alpha"]))
            best_fs = float(best_v["kwargs"].get("flow_sigma", cfg["flow_sigma"]))
            best_msn = float(best_v["kwargs"].get("max_step_norm", cfg["max_step_norm"]))
            kw.update(alpha=best_alpha, flow_sigma=best_fs, max_step_norm=best_msn)
            run_one("syngs", f"combo_a{best_alpha}_fs{best_fs}_msn{best_msn}", **kw)
        except Exception:
            pass
    else:
        print("  No fold-clean improvements found yet.")

    # ── Sweep I: fine alpha (fold-boundary search, current default alpha=0.35 folds) ──────
    # alpha=0.5 → fold=0.006% (just over limit), alpha=0.7 → fold=0.0002% (safe).
    # Search between to find fold-clean high-Dice config.
    print("\n── Sweep I: fine alpha (fold-clean boundary) ──")
    for al in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60]:
        kw = dict(FIXED); kw["alpha"] = al
        run_one("syngs", f"finealpha{al:.2f}", **kw)

    # ── Sweep J: fine msn (with alpha=0.5 near fold boundary) ─────────────────
    # Current msn=0.20 folds at alpha=0.35. Try reducing msn to achieve fold-clean.
    print("\n── Sweep J: fine max_step_norm (with alpha=0.5) ──")
    for msn in [0.16, 0.17, 0.18, 0.19]:
        kw = dict(FIXED); kw["alpha"] = 0.5; kw["max_step_norm"] = msn
        run_one("syngs", f"a0.5_msn{msn}", **kw)

    # ── Sweep K: alpha × msn combos — fold-clean but maximize Dice ────────────
    print("\n── Sweep K: alpha × msn combos ──")
    for al, msn in [(0.40, 0.18), (0.42, 0.19), (0.45, 0.19), (0.48, 0.20), (0.50, 0.25), (0.55, 0.25)]:
        kw = dict(FIXED); kw["alpha"] = al; kw["max_step_norm"] = msn
        run_one("syngs", f"a{al:.2f}_msn{msn:.2f}", **kw)

    # ── Sweep L: bootstrap_mode='none' reference (no jitter augmentation) ─────
    # If 'none' is better than 'antithetic' for these configs, switch default.
    print("\n── Sweep L: bootstrap_mode='none' reference (compare vs antithetic baseline) ──")
    kw = dict(FIXED); kw["bootstrap_mode"] = "none"
    run_one("syngs", "boot_none_default_alpha", **kw)
    kw = dict(FIXED); kw["bootstrap_mode"] = "none"; kw["alpha"] = 0.35; kw["max_step_norm"] = 0.25
    run_one("syngs", "boot_none_msn0.25", **kw)

    # Print summary
    print("\n" + "="*70)
    print("SyNGS SWEEP SUMMARY")
    all_results = load_results()
    gs_results = [(k, v) for k, v in all_results.items()
                  if k.startswith("syngs_") and "error" not in v and v.get("fold_ok")]
    gs_results.sort(key=lambda x: x[1]["dice"], reverse=True)
    print(f"  {'Config':<50s} {'Dice':>8s} {'Δ':>8s} {'Fold%':>8s}")
    for k, v in gs_results[:10]:
        delta = v["dice"] - SYNGS_BASELINE["dice"]
        print(f"  {k:<50s} {v['dice']:>8.5f} {delta:>+8.5f} {v['fold']:>8.5f}%")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tvf", "syngs", "both"], default="both")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed configs")
    args = parser.parse_args()

    print(f"3D Sweep — device={DEVICE}, pair={PAIR_IDX}, fold_limit={FOLD_LIMIT}%")
    print(f"Results → {RESULTS_FILE}")
    print(f"Resume mode: {args.resume}")

    if args.method in ("tvf", "both"):
        sweep_tvf()

    if args.method in ("syngs", "both"):
        sweep_syngs()

    print("\n" + "="*70)
    print("ALL SWEEPS COMPLETE")
    all_results = load_results()
    fold_clean = [(k, v) for k, v in all_results.items()
                  if "error" not in v and v.get("fold_ok") and v["dice"] > 0]
    fold_clean.sort(key=lambda x: x[1]["dice"], reverse=True)
    print(f"\nTop 5 fold-clean results overall:")
    for k, v in fold_clean[:5]:
        baseline = TVF_BASELINE["dice"] if k.startswith("tvf_") else SYNGS_BASELINE["dice"]
        print(f"  {k:<55s} Dice={v['dice']:.5f}  Δ={v['dice']-baseline:+.5f}  fold={v['fold']:.5f}%")
