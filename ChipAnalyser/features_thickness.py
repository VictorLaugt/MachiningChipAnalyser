from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import ColorImage, OpenCVIntArray
    from features_main import MainFeatures


class InsideFeatures:
    ...


class ThicknessAnalysis:
    ...


def extract_inside_features(main_ft: MainFeatures, outside_segments: OpenCVIntArray) -> InsideFeatures:
    ...


def measure_spike_valley_thickness(main_ft: MainFeatures, inside_ft: InsideFeatures) -> ThicknessAnalysis:
    ...


def render_inside_features(render: ColorImage, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
    ...
