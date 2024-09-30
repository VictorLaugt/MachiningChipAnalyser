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
    uncertainty_80 = 1.28 * standard_deviation
    uncertainty_90 = 1.65 * standard_deviation
    uncertainty_95 = 1.96 * standard_deviation

    print(f"Mesure de la grandeur \"{name}\"")
    print(f"droite d'étalonnage: a = {a}, b = {b}")
    print(
        "différence entre les mesures corrigées et les valeurs réelles: "
        f"espérance = {expected_error}, écart-type = {standard_deviation}"
    )
    print(f"incertitude à un niveau de confiance de 80% = {uncertainty_80}")
    print(f"incertitude à un niveau de confiance de 90% = {uncertainty_90}")
    print(f"incertitude à un niveau de confiance de 95% = {uncertainty_95}")
    print()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"étalonnage: {name}")

    calibration_x = np.array((real.min(), real.max()))
    calibration_y = calibration(calibration_x)
    ax.plot(real, measure, 'b+', label="mesures non corrigéesc")
    ax.plot(calibration_x, calibration_y, 'b-')

    ax.plot(real, corrected, 'r+', label="mesures corrigées")
    ax.plot(calibration_x, calibration_x, 'r-')

    ax.set_xlabel("mesure manuelle")
    ax.set_ylabel("mesure automatique")
    ax.legend()
    ax.grid()

    fig, ax = plt.subplots()
    ax.set_title(f"distributions des erreurs de mesure: {name}")
    x = np.linspace(error.min(), error.max(), 1000)
    ax.hist(
        error,
        bins=15,
        density=True,
        align='mid',
        # width=0.7*(x[-1]-x[0])/(15)
    )
    ax.plot(x, 1/(standard_deviation*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x - expected_error)/standard_deviation)**2))

    plt.show()


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
    for ax, name in zip(
        axes,
        ("contact length", "mean peak thickness", "mean valley thickness")
    ):
        plot_measures(ax, auto_measures[name], manu_measures[name], name)
    plt.show()

    # Effectue l'étalonnage de l'appareil de mesure
    for name in ("contact length", "mean peak thickness", "mean valley thickness"):
        auto = auto_measures[name]
        manu = manu_measures[name]
        etalonnage(auto, manu, name)

    plt.show()
