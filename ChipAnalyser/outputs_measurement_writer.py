from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from features_thickness import ThicknessAnalysis

import csv
import numpy as np

import matplotlib.pyplot as plt


class MeasurementWriter:
    """Class for writing measurements into a three column csv file:
    - contact length
    - mean spike thickness
    - mean valley thickness
    """
    header = ("contact length", "mean spike thickness", "mean valley thickness")

    def __init__(self, output_dir: Path, scale: float) -> None:
        self.scale = scale

        self.contact_length_graph = output_dir.joinpath("contact-length-evolution.png")
        self.spike_mean_thk_graph = output_dir.joinpath("spike-mean-thickness-evolution.png")
        self.valley_mean_thk_graph = output_dir.joinpath("valley-mean-thickness-evolution.png")

        self.save_file = output_dir.joinpath("measurements.csv").open(mode='w+', newline='')
        self.csv_writer = csv.writer(self.save_file)
        self.csv_writer.writerow(self.header)

    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        """Write the three measurements as a new line into the csv file."""
        self.csv_writer.writerow((
            self.scale * contact_length,
            self.scale * thickness_analysis.mean_spike_thickness,
            self.scale * thickness_analysis.mean_valley_thickness
        ))

    def write_nan(self) -> None:
        self.csv_writer.writerow((np.nan, np.nan, np.nan))

    def save_graphs(self, display: bool=False) -> None:
        """Save a plot showing the evolution of the three measurements."""
        self.save_file.seek(0)
        csv_reader_itr = iter(csv.reader(self.save_file))
        next(csv_reader_itr)  # skip the header

        contact_length_values = []
        spike_mean_thk_values = []
        valley_mean_thk_values = []
        for contact_length, spike_mean_thk, valley_mean_thk in csv_reader_itr:
            contact_length_values.append(float(contact_length))
            spike_mean_thk_values.append(float(spike_mean_thk))
            valley_mean_thk_values.append(float(valley_mean_thk))
        frames = range(1, len(contact_length_values)+1)

        def new_plot():
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.set_xlabel("frame")
            ax.grid()
            return fig, ax

        fig, ax = new_plot()
        ax.set_title("Contact length evolution")
        ax.set_ylabel("contact length (µm)")
        ax.plot(frames, contact_length_values, '-x')
        fig.savefig(self.contact_length_graph)

        fig, ax = new_plot()
        ax.set_title("Mean spike thickness evolution")
        ax.set_ylabel("mean spike thickness (µm)")
        ax.plot(frames, spike_mean_thk_values, '-x')
        fig.savefig(self.spike_mean_thk_graph)

        fig, ax = new_plot()
        ax.set_title("Mean valley thickness evolution")
        ax.set_ylabel("mean valley thickness (µm)")
        ax.plot(frames, valley_mean_thk_values, '-x')
        fig.savefig(self.valley_mean_thk_graph)

        if display:
            plt.show()

    def release(self) -> None:
        """Close the output csv file."""
        self.save_graphs()
        self.save_file.close()

    def __enter__(self) -> MeasurementWriter:
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_backtrace) -> None:
        self.release()
