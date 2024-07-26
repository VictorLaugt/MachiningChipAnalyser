from pathlib import Path
import csv

import matplotlib.pyplot as plt

import numpy as np
import pywt
import scipy.signal

from experiments.gap_filtering import gap_filter



def extract_signal_from_csv(csv_file_path):
    signals = []
    with csv_file_path.open(mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            signals.append([float(value) for value in row])
    return np.array(signals[0])


def plot_graphs(*signals):
    n = len(signals)

    fig, axes = plt.subplots(n, 1, figsize=(16, 9))
    if n == 1:
        axes = [axes]

    for i, signal in enumerate(signals):
        axes[i].plot(signal, '-x')
        axes[i].grid()

    plt.tight_layout()
    plt.show()


def routine_derivative(csv_path):
    signal = extract_signal_from_csv(csv_path)

    derivative = signal[1:] - signal[:-1]
    derivative_2 = derivative[1:] - derivative[:-1]
    plot_graphs(signal, derivative, derivative_2)


def routine_fft(csv_path):
    signal = extract_signal_from_csv(csv_path)

    coeffs = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1)

    # freq_treshold = 0.5 * (np.mean(np.abs(frequencies)) + np.max(np.abs(frequencies)))
    # freq_treshold = np.mean(np.abs(frequencies))
    freq_treshold = 0.01
    filtered_coeffs = coeffs.copy()
    filtered_coeffs[frequencies > freq_treshold] = 0.

    filtered_signal = np.fft.ifft(filtered_coeffs).real

    plot_graphs(signal, filtered_signal)


def routine_wavelet(csv_path):
    signal = extract_signal_from_csv(csv_path)

    wavelet = 'db4'
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)

    threshold = 20
    filtered_coeffs = [
        c if i == 0 else pywt.threshold(c, threshold, mode='soft')
        for i, c in enumerate(coeffs)
    ]
    filtered_signal = pywt.waverec(filtered_coeffs, wavelet)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(signal, '-x')
    ax.plot(filtered_signal, '-x')
    ax.grid()
    plt.show()


def routine_savgol_peak_characterization(csv_path):
    signal = extract_signal_from_csv(csv_path)
    smoothed = scipy.signal.savgol_filter(signal, window_length=15, polyorder=2)
    rough = scipy.signal.savgol_filter(signal, window_length=45, polyorder=2)

    rough_max, _ = scipy.signal.find_peaks(rough, prominence=5)
    period = np.mean(np.diff(rough_max))

    maximums, _ = scipy.signal.find_peaks(smoothed, distance=0.7*period)
    minimums, _ = scipy.signal.find_peaks(-smoothed, distance=0.7*period, width=0.2*period)

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.scatter(rough_max, rough[rough_max], color='red', marker='o', s=300)
    plt.scatter(maximums, smoothed[maximums], color='black', marker='^', s=500)
    plt.scatter(minimums, smoothed[minimums], color='black', marker='v', s=500)
    ax.plot(signal, '-x', label='signal')
    ax.plot(rough, '-x', label='rough signal')
    ax.plot(smoothed, '-x', label='smoothed signal')
    ax.legend()
    ax.grid()
    plt.show()



for csv_path in (
    Path('results', 'chipcurve', 'thickness_frame_19.csv'),
    Path('results', 'chipcurve', 'thickness_frame_26.csv'),
    Path('results', 'chipcurve', 'thickness_frame_27.csv'),
    Path('results', 'chipcurve', 'thickness_frame_28.csv'),
    Path('results', 'chipcurve', 'thickness_frame_36.csv'),
        ):

    # routine_derivative(csv_path)
    # routine_fft(csv_path)
    # routine_wavelet(csv_path)
    routine_savgol_peak_characterization(csv_path)



