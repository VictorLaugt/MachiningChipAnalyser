import argparse
from pathlib import Path

import csv
import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt


def plot_measures(ax, automatic, manual, title):
    ax.set_title(title)
    ax.plot(automatic, label="automatic measure")
    ax.plot(manual, label="manual measure")
    # ax.set_ylim(0, max(automatic.max(), manual.max()))
    ax.legend()
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


def etalonnage(measure, real, name):
    calibration = Polynomial.fit(real, measure, 1)
    b, a =  calibration.convert().coef
    corrected = (measure - b) / a

    error = real - corrected
    expected_error = np.mean(error)
    standard_deviation = np.sqrt(np.mean((error - expected_error) ** 2))
    uncertainty = {
        80: 1.28 * standard_deviation,
        85: 1.44 * standard_deviation,
        90: 1.645 * standard_deviation,
        95: 1.96 * standard_deviation,
        99: 2.575 * standard_deviation,
    }

    print(f"Mesure de la grandeur \"{name}\"")
    print(f"droite d'étalonnage: a = {a}, b = {b}")
    print(
        "différence entre les mesures corrigées et les valeurs réelles: "
        f"espérance = {expected_error}, écart-type = {standard_deviation}"
    )
    for conf, uncert in uncertainty.items():
        print(f"incertitude à un niveau de confiance de {conf}% = {uncert}")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(name)

    calibration_x = np.array((real.min(), real.max()))
    calibration_y = calibration(calibration_x)
    ax[0].set_title("étalonnage")
    ax[0].plot(real, measure, 'b+', label=f"mesures non corrigées, dispersées autour de y = {a:.3f}x + {b:.3f}")
    ax[0].plot(real, corrected, 'r+', label=f"mesures corrigées, dispersées autour de y = x")
    ax[0].plot(calibration_x, calibration_y, 'b-')
    ax[0].plot(calibration_x, calibration_x, 'r-')
    ax[0].set_xlabel("mesure manuelle")
    ax[0].set_ylabel("mesure automatique")
    ax[0].legend()
    ax[0].grid()

    ax[1].set_title("distribution des erreurs de mesure")
    x = np.linspace(error.min(), error.max(), 1000)
    ax[1].hist(
        error,
        bins=15,
        density=True,
        align='mid',
        color='red',
        alpha=0.3
        # width=0.7*(x[-1]-x[0])/(15)
    )
    ax[1].plot(
        x, gaussian(x, mu=expected_error, sigma=standard_deviation), 'r-',
        label=f"densité de la loi normale centrée d'écart-type {standard_deviation:.3f}",
    )
    ax[1].set_xlabel("différence entre la mesure automatique corrigée et la mesure manuelle")
    ax[1].set_ylabel("densité de probabilité")
    ax[1].legend()
    ax[1].grid()

    plt.show()


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2*np.pi))


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

    # Affiche les valeurs mesurées et les valeurs réelles
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, column_name, name in zip(
        axes,
        ("contact length", "mean peak thickness", "mean valley thickness"),
        ("longueur de contact", "épaisseur moyenne aux pics", "épaisseur moyenne aux vallées")
    ):
        plot_measures(ax, auto_measures[column_name], manu_measures[column_name], name)
    plt.show()

    # Effectue l'étalonnage de l'appareil de mesure
    for column_name, name in zip(
        ("contact length", "mean peak thickness", "mean valley thickness"),
        ("longueur de contact", "épaisseur moyenne aux pics", "épaisseur moyenne aux vallées")
    ):
        etalonnage(auto_measures[column_name], manu_measures[column_name], name)

    plt.show()
