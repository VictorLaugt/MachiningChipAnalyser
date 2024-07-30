from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import FloatArray
    from pathlib import Path
    from measure import ThicknessAnalysis

import abc
import numpy as np

from outputs_graph_animations import GraphAnimator


class AbstractThicknessAnalysisAnimator(abc.ABC):
    @abc.abstractmethod
    def add_frame(self, thk: FloatArray, thk_an: ThicknessAnalysis) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    def __enter__(self) -> AbstractThicknessAnalysisAnimator:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> None:
        self.release()


class NoAnimation(AbstractThicknessAnalysisAnimator):
    def add_frame(self, _thk: FloatArray, _thk_an: ThicknessAnalysis) -> None:
        return

    def release(self) -> None:
        return


class ThicknessAnalysisAnimator(AbstractThicknessAnalysisAnimator):
    def __init__(self, output_dir: Path, scale: float) -> None:
        self.scale = scale
        self.thickness_graph_animation = output_dir.joinpath("chip-thickness-evolution.avi")
        self.thickness_graph_animator = GraphAnimator(
            plot_configs=(
                {'linestyle': '-', 'marker': None, 'label': 'raw thickness measure'},
                {'linestyle': '-', 'marker': None, 'label': 'smoothed thickness'},
                {'linestyle': '-', 'marker': None, 'label': 'rough thickness'}
            ),
            scatter_configs=(
                {'s': 200, 'c': 'red', 'marker': 'o', 'label': 'rough peaks, used to compute the thickness quasiperiod'},
                {'s': 200, 'c': 'black', 'marker': 'v', 'label': 'valleys'},
                {'s': 200, 'c': 'black', 'marker': '^', 'label': 'peaks'}
            ),
            xlabel='inside contour point index',
            ylabel='contact length (µm)',
        )

    def add_frame(self, thk: FloatArray, thk_an: ThicknessAnalysis) -> None:
        thickness = self.scale * thk
        smoothed = self.scale * thk_an.smoothed_thk
        rough = self.scale * thk_an.rough_thk

        rough_spikes = np.column_stack((thk_an.rough_spike_indices, rough[thk_an.rough_spike_indices]))
        valleys = np.column_stack((thk_an.valley_indices, smoothed[thk_an.valley_indices]))
        spikes = np.column_stack((thk_an.spike_indices, smoothed[thk_an.spike_indices]))

        self.thickness_graph_animator.add_frame(
            plot_ydata=(thickness, smoothed, rough),
            scatter_data=(rough_spikes, valleys, spikes)
        )

    def release(self):
        self.thickness_graph_animator.save(self.thickness_graph_animation)
