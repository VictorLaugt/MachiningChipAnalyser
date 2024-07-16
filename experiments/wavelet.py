import numpy as np
import matplotlib.pyplot as plt
import pywt

from pathlib import Path
import csv

signals = []

csv_file_path = Path('results', 'chipcurve', 'thickness.csv')
with csv_file_path.open(mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        signals.append([float(value) for value in row])


signal = signals[0]


# Décomposer le signal en utilisant les ondelettes
wavelet = 'db4'  # Choisir une ondelette (Daubechies 4 dans ce cas)
coeffs = pywt.wavedec(signal, wavelet, level=6)

# Seuiling (soft thresholding) pour débruiter
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
threshold = 10 * sigma * np.sqrt(2 * np.log(len(signal)))
# coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
coeffs_thresholded = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]

# Reconstituer le signal débruité
signal_denoised = pywt.waverec(coeffs_thresholded, wavelet)

# Afficher les résultats
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(signal, label='signal bruité')
ax.plot(signal_denoised, label='signal débruité')
ax.grid(True)
plt.show()

