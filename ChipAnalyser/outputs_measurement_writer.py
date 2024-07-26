from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from features_thickness import ThicknessAnalysis

import abc
import csv

import matplotlib.pyplot as plt


class AbstractMeasurementWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


class MeasurementPrinter(AbstractMeasurementWriter):
    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        ...
        # print(
        #     f"contact length = {contact_length}, "
        #     f"spike mean thickness = {spike_mean_thickness}, "
        #     f"valley mean thickness = {valley_mean_thickness}"
        # )
        print(f"contact length = {contact_length}")

    def release(self) -> None:
        return


class MeasurementWriter(AbstractMeasurementWriter):
    header = ("contact length", "spike mean thickness", "valley mean thickness")

    def __init__(self, output_dir: Path, scale: float) -> None:
        self.scale = scale

        self.contact_length_graph = output_dir.joinpath("contact-length-evolution.png")
        self.spike_mean_thk_graph = output_dir.joinpath("spike-mean-thickness-evolution.png")
        self.valley_mean_thk_graph = output_dir.joinpath("valley-mean-thickness-evolution.png")

        self.save_file = output_dir.joinpath("measurements.csv").open(mode='w+')
        self.csv_writer = csv.writer(self.save_file)
        self.csv_writer.writerow(self.header)

    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        spike_mean_thickness = valley_mean_thickness = 0.0  # MOCK mean thickness value to write into the csv file
        self.csv_writer.writerow((
            self.scale * contact_length,
            self.scale * spike_mean_thickness,
            self.scale * valley_mean_thickness
        ))

    def save_graphs(self, display: bool=False) -> None:
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
        ax.set_title("Spike mean thickness evolution")
        ax.set_ylabel("spike mean thickness (µm)")
        ax.plot(frames, spike_mean_thk_values, '-x')
        fig.savefig(self.spike_mean_thk_graph)

        fig, ax = new_plot()
        ax.set_title("Valley mean thickness evolution")
        ax.set_ylabel("spike mean thickness (µm)")
        ax.plot(frames, valley_mean_thk_values, '-x')
        fig.savefig(self.valley_mean_thk_graph)

        if display:
            plt.show()

    def release(self) -> None:
        self.save_graphs()
        self.save_file.close()
