import argparse
from pathlib import Path

from dataclasses import dataclass

import csv
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial import Polynomial


def comp_graph(ax, automatic, manual, title):
    ax.set_title(title)
    ax.plot(automatic, label="automatic measure")
    ax.plot(manual, label="manual measure")
    # ax.set_ylim(0, max(automatic.max(), manual.max()))
    ax.legend()
    ax.grid()


def courbe_etalonnage(ax, automatic, manual, name):
    ax.set_title(f"courbe d'étalonnage: {name}")

    ax.plot(manual, automatic, '+')

    linear_reg = Polynomial.fit(manual, automatic, 1)
    linear_reg_x = np.array((manual.min(), manual.max()))
    linear_reg_y = linear_reg(linear_reg_x)
    ax.plot(linear_reg_x, linear_reg_y, '-')

    # ax.set_xlim(0, manual.max())
    # ax.set_ylim(0, automatic.max())

    ax.set_xlabel("mesure manuelle")
    ax.set_ylabel("mesure automatique")
    ax.grid()


def read_measures(measure_file_path):
    contact_len = []
    mean_peak_thk = []
    mean_valley_thk = []

    with measure_file_path.open(mode='r') as measure_file:
        reader = csv.DictReader(measure_file)
        for row in reader:
            contact_len.append(row["contact length"])
            mean_peak_thk.append(row["mean peak thickness"])
            mean_valley_thk.append(row["mean valley thickness"])

    return {
        "contact length": np.array(contact_len, dtype=float),
        "mean peak thickness": np.array(mean_peak_thk, dtype=float),
        "mean valley thickness": np.array(mean_valley_thk, dtype=float)
    }


if __name__ == '__main__':
    def existing_directory(arg):
        try:
            input_dir = Path(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid path: {arg}")
        if not input_dir.is_dir():
            raise argparse.ArgumentTypeError(f"directory not found: {arg}")
        return input_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=existing_directory)
    args = parser.parse_args()

    auto_measures = read_measures(args.input_dir.joinpath('automatic_measures.csv'))
    manu_measures = read_measures(args.input_dir.joinpath('manual_measures.csv'))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, name in zip(
        axes,
        ("contact length", "mean peak thickness", "mean valley thickness")
    ):
        comp_graph(ax, auto_measures[name], manu_measures[name], name)
    plt.show()

    for name in ("contact length", "mean peak thickness", "mean valley thickness"):
        fig, ax = plt.subplots(figsize=(10, 10))
        auto = auto_measures[name]
        manu = manu_measures[name]
        courbe_etalonnage(ax, auto, manu, name)

        erreur_absolue = np.mean(auto - manu)
        erreur_relative = np.mean(erreur_absolue / np.abs(manu))

        print(f"\n{name}:")
        print(f"erreur absolue = {erreur_absolue}")
        print(f"erreur relative = {erreur_relative}")

    plt.show()

