from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence
    from type_hints import FloatArray, FloatPtArray
    from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def padded_array(array: np.ndarray, padded_length: int) -> np.ndarray:
    padded_array = np.full(padded_length, np.nan, array.dtype)
    padded_array[:len(array)] = array
    return padded_array


class SaveAnimation:
    def __init__(self, graph_animator: GraphAnimator, file_path: Path, display: bool=False) -> None:
        self.data = graph_animator

        fig, ax = plt.subplots(figsize=(16, 9))
        self.frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        self.plots = []
        self.scatters = []
        self.artists = [self.frame_text]

        for plot_idx in range(len(self.data.plot_configs)):
            ydata = padded_array(self.data.plot_ydata[0][plot_idx], self.data.plot_ydata_length[plot_idx])
            config = self.data.plot_configs[plot_idx]
            plot = ax.plot(ydata, **config)[0]
            self.plots.append(plot)
            self.artists.append(plot)
        for scatter_idx in range(len(self.data.scatter_configs)):
            pts = self.data.scatter_data[0][scatter_idx]
            config = self.data.scatter_configs[scatter_idx]
            scatter = ax.scatter(pts[:, 0], pts[:, 1], **config)
            self.scatters.append(scatter)
            self.artists.append(scatter)
        ax.set_xlabel(self.data.xlabel)
        ax.set_ylabel(self.data.ylabel)
        ax.grid()
        ax.legend(loc='upper right')

        animation = anim.FuncAnimation(
            fig,
            self.update_animation,
            frames=len(self.data.plot_ydata),
            interval=1000/30,
            blit=True
        )
        animation.save(file_path)  # TODO: try different .save() parameters to get rid of the strange artefacts on windows
        if display:
            plt.show()

    def update_animation(self, frame_idx: int) -> Iterable[plt.Artist]:
        for plot_idx in range(len(self.data.plot_configs)):
            ydata = padded_array(self.data.plot_ydata[frame_idx][plot_idx], self.data.plot_ydata_length[plot_idx])
            self.plots[plot_idx].set_ydata(ydata)
        for scatter_idx in range(len(self.data.scatter_configs)):
            self.scatters[scatter_idx].set_offsets(self.data.scatter_data[frame_idx][scatter_idx])
        self.frame_text.set_text(f'frame: {frame_idx+1}')
        return self.artists


class GraphAnimator:
    """
    for (0 <= j < plot_nb):
        plot_ydata_length == max(plot_ydata[i][j] for 0 <= i < frame_nb)
    for (0 <= i < frame_nb; 0 <= j < plot_nb):
        plot_ydata[i][j].shape == (point_nb[j],)
    for (0 <= i < frame_nb; 0 <= j < scatter_nb):
        scatter_data[i][j].shape == (scatter_point_nb[i, j], 2)
    """
    def __init__(
        self,
        plot_configs: Sequence[dict],
        scatter_configs: Sequence[dict],
        xlabel: str,
        ylabel: str
    ) -> None:
        self.plot_configs = plot_configs
        self.scatter_configs = scatter_configs

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.plot_ydata_length = [0] * len(plot_configs)
        self.plot_ydata = []
        self.scatter_data = []

    def append_frame(self, plot_ydata: Sequence[FloatArray], scatter_data: Sequence[FloatPtArray]) -> None:
        assert len(plot_ydata) == len(self.plot_configs) and len(scatter_data) == len(self.scatter_configs)
        for i, ydata in enumerate(plot_ydata):
            self.plot_ydata_length[i] = max(self.plot_ydata_length[i], len(ydata))
        self.plot_ydata.append(plot_ydata)
        self.scatter_data.append(scatter_data)


    def save(self, file_path: Path) -> None:
        if len(self.plot_ydata) > 0:
            _ = SaveAnimation(self, file_path)
