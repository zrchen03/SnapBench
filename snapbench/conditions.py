"""Controlled corruption conditions used in SnapBench.

The paper reports **53 corruption conditions** (45 image + 8 text).
Adding the clean query yields **54 evaluation states**.
"""

from __future__ import annotations

TEXT_OPERATORS = (
    "char_add",
    "char_delete",
    "char_change",
    "char_swap",
    "word_repeat",
    "word_swap",
    "sent_add",
    "sent_replace",
)

IMAGE_OPERATORS = (
    "low_light",
    "overexposure",
    "defocus_blur",
    "motion_blur",
    "compression",
    "low_resolution",
    "rotation",
    "perspective",
    "lens_distortion",
    "cropping",
    "downscale",
    "watermark",
    "mosaic",
    "scribble",
    "ui_elements",
)

IMAGE_PRIMITIVES = {
    "Degrade": ("defocus_blur", "motion_blur", "low_light", "overexposure", "low_resolution", "compression"),
    "Transform": ("rotation", "perspective", "lens_distortion"),
    "Remove": ("cropping", "downscale"),
    "Add": ("watermark", "mosaic", "scribble", "ui_elements"),
}

SEVERITIES = (1, 2, 3)

N_IMAGE_CONDITIONS = len(IMAGE_OPERATORS) * len(SEVERITIES)  # 45
N_TEXT_CONDITIONS = len(TEXT_OPERATORS)  # 8
N_CORRUPTION_CONDITIONS = N_IMAGE_CONDITIONS + N_TEXT_CONDITIONS  # 53
N_EVAL_STATES = N_CORRUPTION_CONDITIONS + 1  # 54, including clean


def image_condition_name(operator: str, severity: int) -> str:
    if operator not in IMAGE_OPERATORS:
        raise ValueError(f"unknown image operator: {operator}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return f"{operator}/sev{severity}"


def all_image_conditions() -> list[str]:
    return [image_condition_name(op, sev) for op in IMAGE_OPERATORS for sev in SEVERITIES]


def all_conditions(*, include_clean: bool = True) -> list[str]:
    names = ["clean"] if include_clean else []
    names.extend(TEXT_OPERATORS)
    names.extend(all_image_conditions())
    return names
