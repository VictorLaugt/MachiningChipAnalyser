from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path

import abc
import csv


class AbstractMeasurementWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, contact_length: float, spike_mean_thickness: float, valley_mean_thickness: float) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass


class MeasurementPrinter(AbstractMeasurementWriter):
    def write(self, contact_length: float, spike_mean_thickness: float, valley_mean_thickness: float) -> None:
        print(
            f"contact length = {contact_length}, "
            f"spike mean thickness = {spike_mean_thickness}, "
            f"valley mean thickness = {valley_mean_thickness}"
        )

    def release(self) -> None:
        return


class MeasurementWriter(AbstractMeasurementWriter):
    header = ("contact length", "spike mean thickness", "valley mean thickness")

    def __init__(self, data_file_path: Path) -> None:
        self.save_file = data_file_path.open(mode='w')
        self.csv_writer = csv.writer(data_file_path)
        self.csv_writer.writerow(self.header)

    def write(self, contact_length: float, spike_mean_thickness: float, valley_mean_thickness: float) -> None:
        self.csv_writer.writerow((contact_length, spike_mean_thickness, valley_mean_thickness))

    def release(self) -> None:
        self.save_file.close()
