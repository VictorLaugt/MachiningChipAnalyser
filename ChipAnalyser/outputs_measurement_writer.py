from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
    from features_thickness import ThicknessAnalysis

import abc
import csv


class AbstractMeasurementWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


# TODO: MeasurementPrinter and MeasurementWriter
class MeasurementPrinter(AbstractMeasurementWriter):
    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        ...
        # print(
        #     f"contact length = {contact_length}, "
        #     f"spike mean thickness = {spike_mean_thickness}, "
        #     f"valley mean thickness = {valley_mean_thickness}"
        # )

    def release(self) -> None:
        return


class MeasurementWriter(AbstractMeasurementWriter):
    header = ("contact length", "spike mean thickness", "valley mean thickness")

    def __init__(self, data_file_path: Path, scale: float) -> None:
        self.save_file = data_file_path.open(mode='w')
        self.scale = scale
        self.csv_writer = csv.writer(self.save_file)
        self.csv_writer.writerow(self.header)

    def write(self, contact_length: float, thickness_analysis: ThicknessAnalysis) -> None:
        spike_mean_thickness = valley_mean_thickness = 0.0  # MOCK mean thickness value to write into the csv file
        self.csv_writer.writerow((
            self.scale * contact_length,
            self.scale * spike_mean_thickness,
            self.scale * valley_mean_thickness
        ))

    def release(self) -> None:
        self.save_file.close()
