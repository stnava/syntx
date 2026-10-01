"""
syntx.policy — rule table that maps a ``PairDiagnosis`` to registration settings.

``synthesize_policy`` returns a ``RegistrationPolicy`` (transform type, similarity metric,
guidance, denoising, initial-alignment mode, CT intensity window, plus regulariser numbers and
a text ``explanation``). The rules are fixed if/else choices, not tuned or learned.

``syntx.auto_reg`` reads only some of the fields, and only where the caller did not set the
corresponding argument: ``transform_type`` (and ``guided``, only when the transform type is
taken from the policy), ``denoise``, ``cohort_type``, ``robust_affine``,
``similarity_metric`` (forwarded as a keyword) and ``ct_window`` (applied when the fixed image
is diagnosed CT). ``regularizer``, ``sobolev_alpha``, ``dsti_alpha``, ``grad_step``,
``flow_sigma``, ``total_sigma``, ``guided_weight`` and ``parameters`` are not used by
``auto_reg``; ``explanation`` is only printed / stored.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
from .diagnose import PairDiagnosis, ImageDiagnosis, diagnose_pair


@dataclass
class RegistrationPolicy:
    """Registration settings chosen by ``synthesize_policy``.

    Attributes
    ----------
    transform_type : str, default "SyN"
        "SyN" or "TVF" in the current rules.
    similarity_metric : str, default "cc2"
        "cc2" or "mattes_mi".
    regularizer : str, default "sobolev"
        "sobolev" or "dsti1" (descriptive; not applied by ``auto_reg``).
    sobolev_alpha, dsti_alpha : float, defaults 1.5, 0.035
        Regulariser strengths (not applied by ``auto_reg``; ``dsti_alpha`` is not in
        ``to_dict``).
    grad_step, flow_sigma, total_sigma : float, defaults 0.25, 5.0, 0.0
        Optimiser / smoothing values (not applied by ``auto_reg``).
    guided : str or None, default None
        "sulcal" for whole-brain mono-modal MRI, else None.
    guided_weight : float or None, default None
        Never set by the rules.
    cohort_type : str, default "auto"
        Never changed by the rules.
    robust_affine : bool or str, default "auto"
        Initial-alignment mode ("auto" or "translation_only").
    denoise : bool, default False
        Request Rician non-local-means denoising (``auto_reg``: 3-D only, needs antstorch).
    ct_window : (float, float) or None
        HU window (min, max) mapped linearly to [0, 1] and clipped.
    explanation : str
        Human-readable reason for the choice.
    parameters : dict, default {}
        Extra entries merged into ``to_dict``; never filled by the rules.
    """
    transform_type: str = "SyN"
    similarity_metric: str = "cc2"
    regularizer: str = "sobolev"
    sobolev_alpha: float = 1.5
    dsti_alpha: float = 0.035
    grad_step: float = 0.25
    flow_sigma: float = 5.0
    total_sigma: float = 0.0
    guided: Optional[str] = None
    guided_weight: Optional[float] = None
    cohort_type: str = "auto"
    robust_affine: Union[bool, str] = "auto"
    denoise: bool = False
    ct_window: Optional[Tuple[float, float]] = None
    explanation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the settings as a dict keyed like registration keywords.

        Keys: ``type_of_transform``, ``similarity_metric``, ``regularizer``,
        ``sobolev_alpha``, ``grad_step``, ``flow_sigma``, ``total_sigma``, ``guided``,
        ``guided_weight``, ``cohort_type``, ``robust_affine``, ``denoise``, ``explanation``,
        ``ct_window`` (only when set), then ``parameters`` merged on top. ``dsti_alpha`` is not
        included. Not every key is accepted by every registration function (e.g.
        ``tvf_registration`` rejects ``similarity_metric`` and ``sobolev_alpha``), and
        ``auto_reg`` has no ``ct_window`` / ``explanation`` / ``regularizer`` arguments (it
        would forward them to the registration function), so the dict cannot be passed to
        ``auto_reg(**d)`` as is.
        """
        d = {
            "type_of_transform": self.transform_type,
            "similarity_metric": self.similarity_metric,
            "regularizer": self.regularizer,
            "sobolev_alpha": self.sobolev_alpha,
            "grad_step": self.grad_step,
            "flow_sigma": self.flow_sigma,
            "total_sigma": self.total_sigma,
            "guided": self.guided,
            "guided_weight": self.guided_weight,
            "cohort_type": self.cohort_type,
            "robust_affine": self.robust_affine,
            "denoise": self.denoise,
            "explanation": self.explanation,
        }
        if self.ct_window is not None:
            d["ct_window"] = self.ct_window
        d.update(self.parameters)
        return d


def synthesize_policy(pair_diag: PairDiagnosis) -> RegistrationPolicy:
    """
    Pick registration settings from a pair diagnosis with a fixed rule list (first match wins).

    1. Both BRAIN: same modality -> SyN, cc2, ``guided="sulcal"``, ``robust_affine="auto"``,
       ``denoise=True`` (no guidance and "translation_only" alignment if either diagnosis has
       ``details["is_roi_crop"]``, which ``diagnose_image`` never sets); different modality ->
       SyN, mattes_mi, ``denoise=True``.
    2. Either THORAX and both CT -> TVF, cc2, regulariser "dsti1" (alpha 0.035), grad_step
       0.5, flow_sigma 1.0, ``ct_window=(-1000, 400)``.
    3. Either ABDOMEN and both CT -> SyN, cc2, ``ct_window=(-150, 250)``.
    4. Either HEART (only the classifier produces it) -> SyN, cc2, ``denoise=True``.
    5. Either PELVIS -> SyN, cc2, ``sobolev_alpha=2.0``.
    6. ``relationship == "CROSS_MODAL"`` -> SyN, mattes_mi.
    7. Otherwise -> SyN, cc2.

    Parameters
    ----------
    pair_diag : PairDiagnosis
        From ``syntx.diagnose.diagnose_pair``.

    Returns
    -------
    RegistrationPolicy
        Fields not listed above keep the dataclass defaults. The ``explanation`` strings
        describe intent and mention features (e.g. specific affine strategies) that this
        function does not configure.
    """
    fix = pair_diag.fixed
    mov = pair_diag.moving

    # Rule 1: Brain MRI to Brain MRI
    if fix.is_brain() and mov.is_brain():
        is_roi = fix.details.get("is_roi_crop", False) or mov.details.get("is_roi_crop", False)
        guided = None if is_roi else "sulcal"
        affine_mode = "translation_only" if is_roi else "auto"

        if pair_diag.is_same_modality:
            # Mono-modal Brain MRI (e.g. T1 to T1)
            return RegistrationPolicy(
                transform_type="SyN",
                similarity_metric="cc2",
                regularizer="sobolev",
                sobolev_alpha=1.5,
                grad_step=0.25,
                guided=guided,
                cohort_type="auto",
                robust_affine=affine_mode,
                denoise=True,
                explanation=(
                    f"Diagnosed Brain MRI (Mono-modal, {'sub-structural ROI crop' if is_roi else 'whole brain'}): "
                    f"Configured Eulerian Sobolev SyN with cc2, {'translation-only' if is_roi else 'deterministic 18-cone'} "
                    f"robust affine pre-alignment, adaptive Rician denoising, and "
                    f"{'disabled sulcal guidance for sub-structural ROI' if is_roi else 'turnkey Sulcal Soft Dice guidance'}."
                )
            )
        else:
            # Multi-contrast Brain MRI (e.g. T1 to T2 or FLAIR)
            return RegistrationPolicy(
                transform_type="SyN",
                similarity_metric="mattes_mi",
                regularizer="sobolev",
                sobolev_alpha=1.5,
                grad_step=0.25,
                guided=None,
                robust_affine="auto",
                denoise=True,
                explanation=(
                    "Diagnosed Brain MRI (Multi-contrast): Configured Mattes Mutual Information "
                    "with boundary padding to maximize joint entropy across contrasting modalities, "
                    "coupled with Sobolev SyN and Rician denoising."
                )
            )

    # Rule 2: Thorax / Lung CT
    if fix.is_thorax() or mov.is_thorax():
        if fix.is_ct() and mov.is_ct():
            return RegistrationPolicy(
                transform_type="TVF",
                similarity_metric="cc2",
                regularizer="dsti1",
                dsti_alpha=0.035,
                grad_step=0.50,
                flow_sigma=1.0,
                robust_affine="auto",
                denoise=False,
                ct_window=(-1000.0, 400.0),
                explanation=(
                    "Diagnosed Thorax CT (Lung): Configured Continuous Time-Varying Velocity Field (TVF) "
                    "with Dirichlet shield (dsti1) to absorb extreme non-rigid respiratory deformation, "
                    "lung HU windowing [-1000, 400], and robust multi-start affine initialization."
                )
            )

    # Rule 3: Abdomen CT (Liver, Spleen, Pancreas, Colon)
    if fix.is_abdomen() or mov.is_abdomen():
        if fix.is_ct() and mov.is_ct():
            return RegistrationPolicy(
                transform_type="SyN",
                similarity_metric="cc2",
                regularizer="sobolev",
                sobolev_alpha=1.5,
                grad_step=0.25,
                robust_affine="auto",
                denoise=False,
                ct_window=(-150.0, 250.0),
                explanation=(
                    "Diagnosed Abdomen CT: Configured abdominal soft tissue HU windowing [-150, 250] "
                    "with Eulerian Sobolev SyN and robust multi-start affine pre-alignment."
                )
            )

    # Rule 4: Cardiac MRI
    if fix.is_heart() or mov.is_heart():
        return RegistrationPolicy(
            transform_type="SyN",
            similarity_metric="cc2",
            regularizer="sobolev",
            sobolev_alpha=1.5,
            grad_step=0.25,
            guided=None,
            robust_affine="auto",
            denoise=True,
            explanation=(
                "Diagnosed Cardiac MRI: Configured Eulerian Sobolev SyN with squared cross-correlation, "
                "SE(3) diverse robust affine initialization, and adaptive Rician denoising."
            )
        )

    # Rule 5: Pelvis / Prostate MRI
    if fix.is_pelvis() or mov.is_pelvis():
        return RegistrationPolicy(
            transform_type="SyN",
            similarity_metric="cc2",
            regularizer="sobolev",
            sobolev_alpha=2.0,
            grad_step=0.25,
            guided=None,
            robust_affine="auto",
            denoise=False,
            explanation=(
                "Diagnosed Pelvis MRI (Prostate): Configured Eulerian Sobolev SyN with cc2, "
                "anisotropy-regularized robust affine initialization for thick-slice slabs, "
                "and disabled sulcal guidance."
            )
        )

    # Rule 5: Cross-Modality (e.g. MRI to CT)
    if pair_diag.relationship == "CROSS_MODAL":
        return RegistrationPolicy(
            transform_type="SyN",
            similarity_metric="mattes_mi",
            regularizer="sobolev",
            sobolev_alpha=1.5,
            grad_step=0.25,
            guided=None,
            robust_affine="auto",
            denoise=False,
            explanation=(
                "Diagnosed Cross-Modality (MRI <-> CT): Enforced Mattes Mutual Information "
                "with Parzen boundary-padded windowing for non-monotonic joint intensity registration."
            )
        )

    # Rule 6: General Default
    return RegistrationPolicy(
        transform_type="SyN",
        similarity_metric="cc2",
        regularizer="sobolev",
        sobolev_alpha=1.5,
        grad_step=0.25,
        robust_affine="auto",
        denoise=False,
        explanation="General Fallback: Configured Eulerian Sobolev SyN with squared cross-correlation."
    )


def auto_policy_for_images(fixed, moving, fast: bool = True) -> RegistrationPolicy:
    """Return ``synthesize_policy(diagnose_pair(fixed, moving, fast=fast))``.

    ``fast=True`` (default) uses the heuristic diagnosis only (no classifier).
    """
    pair_diag = diagnose_pair(fixed, moving, fast=fast)
    return synthesize_policy(pair_diag)
