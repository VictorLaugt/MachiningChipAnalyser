import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import csv


def extract_signal_from_csv(csv_file_path):
    signals = []
    with csv_file_path.open(mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            signals.append([float(value) for value in row])
    return np.array(signals[0])


def plot_signal(*signals):
    n = len(signals)

    fig, axes = plt.subplots(n, 1, figsize=(16, 9))
    if n == 1:
        axes = [axes]

    for i, signal in enumerate(signals):
        axes[i].plot(signal, '-x')
        axes[i].grid()

    plt.tight_layout()
    plt.show()


def routine(csv_path):
    signal = extract_signal_from_csv(csv_path)

    derivative = signal[1:] - signal[:-1]
    derivative_2 = derivative[1:] - derivative[:-1]
    plot_signal(signal, derivative, derivative_2)


for csv_path in (
    Path('results', 'chipcurve', 'thickness_frame_19.csv'),
    Path('results', 'chipcurve', 'thickness_frame_26.csv'),
    Path('results', 'chipcurve', 'thickness_frame_27.csv'),
    Path('results', 'chipcurve', 'thickness_frame_28.csv'),
    Path('results', 'chipcurve', 'thickness_frame_36.csv'),
        ):
    routine(csv_path)



