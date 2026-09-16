"""
syntx.policy — Autonomous Registration Policy Synthesis
=======================================================

Synthesizes Pareto-optimal registration recipes from image diagnoses:
- Model selection (SyN Eulerian vs TVF Dirichlet-Shield vs Affine)
- Similarity functional (CC2 vs Mattes MI vs Deep LNCC)
- Spatial regularizer (Sobolev alpha=1.5 vs DSTI-1 alpha=0.035)
- Geometric guidance (Sulcal Soft Dice vs Pleural vs None)
- Adaptive preprocessing (CT HU windowing vs MRI N4/Rician denoising)
- Affine initialization (18-cone rotational multi-start vs CoM translation)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
from .diagnose import PairDiagnosis, ImageDiagnosis, diagnose_pair


@dataclass
class RegistrationPolicy:
    """Executable registration configuration synthesized by the autonomous policy engine."""
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
        """Returns dictionary representation suitable for auto_reg kwargs."""
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
    Synthesizes an optimal registration plan based on diagnosed modality and anatomy.

    Parameters:
    -----------
    pair_diag : PairDiagnosis
        Joint diagnostic assessment of fixed and moving images.

    Returns:
    --------
    RegistrationPolicy
        Verified registration policy ready for execution by auto_reg.
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
    """Convenience helper to diagnose images and synthesize registration policy in one step."""
    pair_diag = diagnose_pair(fixed, moving, fast=fast)
    return synthesize_policy(pair_diag)
