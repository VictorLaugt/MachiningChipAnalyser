from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from features_main import MainFeatures


class InsideFeatures:
    ...


class ThicknessAnalysis:
    ...


def measure_spike_valley_thickness(main_ft: MainFeatures, inside_ft: InsideFeatures) -> ThicknessAnalysis:
    ...
