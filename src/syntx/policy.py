"""
syntx.policy — rule table that maps a ``PairDiagnosis`` to registration settings.

``synthesize_policy`` returns a ``RegistrationPolicy``: transform type, similarity metric,
guidance, denoising, initial-alignment mode and CT intensity window, plus a text
``explanation``. The rules are fixed if/else choices, not tuned or learned. Every field is
applied by ``syntx.auto_reg`` (where the caller did not set the corresponding argument);
regularisation and optimiser settings are left to each method's tuned defaults.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
from .diagnose import PairDiagnosis, ImageDiagnosis, diagnose_pair


@dataclass
class RegistrationPolicy:
    """Registration settings chosen by ``synthesize_policy`` (all applied by ``auto_reg``).

    Attributes
    ----------
    transform_type : str, default "SyN"
        "SyN" or "TVF" in the current rules (``auto_reg``'s ``type_of_transform``).
    similarity_metric : str, default "cc2"
        "cc", "cc2" or "mattes_mi" (forwarded as ``syn_metric``).
    guided : str or None, default None
        "sulcal" for whole-brain mono-modal MRI, else None.
    cohort_type : str, default "auto"
        Never changed by the rules.
    robust_affine : bool or str, default "auto"
        Initial-alignment mode (the rules always use "auto").
    denoise : bool, default False
        Request Rician non-local-means denoising (``auto_reg``: 3-D only, needs antstorch).
    ct_window : (float, float) or None
        HU window (min, max) mapped linearly to [0, 1] and clipped (applied when the fixed
        image is diagnosed CT).
    coarse_spacing_mm : float, default 4.0
        Target coarsest isotropic pyramid spacing in millimeters for scale-space level planning
        in ``auto_reg`` (4.0 mm for cranial/pelvic/general; 10.0 mm for thoraco-abdominal
        respiratory/visceral cavities).
    explanation : str
        Human-readable reason for the choice.
    """
    transform_type: str = "SyN"
    similarity_metric: str = "cc2"
    guided: Optional[str] = None
    cohort_type: str = "auto"
    robust_affine: Union[bool, str] = "auto"
    denoise: bool = False
    ct_window: Optional[Tuple[float, float]] = None
    coarse_spacing_mm: float = 4.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """The settings as ``auto_reg`` keywords: ``type_of_transform``, ``syn_metric``,
        ``guided``, ``cohort_type``, ``robust_affine``, ``denoise`` (``ct_window`` and
        ``explanation`` are not ``auto_reg`` arguments and are left out), so
        ``auto_reg(fixed, moving, diagnose=False, **policy.to_dict())`` reproduces the policy
        apart from CT windowing."""
        return {
            "type_of_transform": self.transform_type,
            "syn_metric": self.similarity_metric,
            "guided": self.guided,
            "cohort_type": self.cohort_type,
            "robust_affine": self.robust_affine,
            "denoise": self.denoise,
        }


def synthesize_policy(pair_diag: PairDiagnosis) -> RegistrationPolicy:
    """
    Pick registration settings from a pair diagnosis with a fixed rule list (first match wins).

    1. Both BRAIN: same modality -> SyN, cc2, ``guided="sulcal"``, ``denoise=True``;
       different modality -> SyN, mattes_mi, ``denoise=True``.
    2. Fixed THORAX and both CT -> TVF, cc (signed LNCC), ``ct_window=(-1000, 400)``,
       ``coarse_spacing_mm=10.0``.
    3. Either ABDOMEN and both CT -> SyN, cc (signed LNCC), ``ct_window=(-150, 250)``,
       ``coarse_spacing_mm=10.0``.
    4. Either HEART (only the classifier produces it) -> SyN, cc2, ``denoise=True``,
       ``coarse_spacing_mm=10.0``.
    5. Either PELVIS -> SyN, cc2 (``denoise=True`` for 3-D MRI).
    6. ``relationship == "CROSS_MODAL"`` -> SyN, mattes_mi.
    7. Otherwise -> SyN, cc2 (``denoise=True`` for 3-D MRI).

    Parameters
    ----------
    pair_diag : PairDiagnosis
        From ``syntx.diagnose.diagnose_pair``.

    Returns
    -------
    RegistrationPolicy
        Fields not listed above keep the dataclass defaults (``robust_affine="auto"``).
    """
    fix = pair_diag.fixed
    mov = pair_diag.moving

    if fix.is_brain() and mov.is_brain():
        if pair_diag.is_same_modality:
            return RegistrationPolicy(
                transform_type="SyN", similarity_metric="cc2", guided="sulcal", denoise=True,
                explanation="Brain MRI, same modality: SyN, cc2, robust affine, Rician denoising, "
                            "sulcal guidance.")
        return RegistrationPolicy(
            transform_type="SyN", similarity_metric="mattes_mi", denoise=True,
            explanation="Brain MRI, different contrasts: SyN, Mattes MI, Rician denoising.")

    if fix.is_thorax() and fix.is_ct() and mov.is_ct():
        return RegistrationPolicy(
            transform_type="TVF", similarity_metric="cc", ct_window=(-1000.0, 400.0),
            coarse_spacing_mm=10.0,
            explanation="Thorax CT: TVF, signed cc, lung HU window [-1000, 400], 10 mm coarse scale.")

    if (fix.is_abdomen() or mov.is_abdomen()) and fix.is_ct() and mov.is_ct():
        return RegistrationPolicy(
            transform_type="SyN", similarity_metric="cc", ct_window=(-150.0, 250.0),
            coarse_spacing_mm=10.0,
            explanation="Abdomen CT: SyN, signed cc, soft-tissue HU window [-150, 250], 10 mm coarse scale.")

    if fix.is_heart() or mov.is_heart():
        return RegistrationPolicy(
            transform_type="SyN", similarity_metric="cc2", denoise=True,
            coarse_spacing_mm=10.0,
            explanation="Cardiac MRI: SyN, cc2, Rician denoising, 10 mm coarse scale.")

    denoise_3d_mri = bool(fix.is_mri() and mov.is_mri() and fix.dimension >= 3)

    if fix.is_pelvis() or mov.is_pelvis():
        return RegistrationPolicy(
            transform_type="SyN", similarity_metric="cc2",
            guided="mind" if denoise_3d_mri else None,
            denoise=denoise_3d_mri,
            explanation="Pelvis: SyN, cc2" + (", Rician denoising, MIND self-similarity guidance." if denoise_3d_mri else "."))

    if pair_diag.relationship == "CROSS_MODAL":
        return RegistrationPolicy(
            transform_type="SyN", similarity_metric="mattes_mi",
            explanation="Cross-modality: SyN, Mattes MI.")

    return RegistrationPolicy(transform_type="SyN", similarity_metric="cc2",
                              denoise=denoise_3d_mri,
                              explanation="Default: SyN, cc2.")

def auto_policy_for_images(fixed, moving, fast: bool = True) -> RegistrationPolicy:
    """Return ``synthesize_policy(diagnose_pair(fixed, moving, fast=fast))``.

    ``fast=True`` (default) uses the heuristic diagnosis only (no classifier).
    """
    pair_diag = diagnose_pair(fixed, moving, fast=fast)
    return synthesize_policy(pair_diag)
