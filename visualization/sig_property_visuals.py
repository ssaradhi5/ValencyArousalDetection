import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from processing.build_features import signal_properties
from pyedflib import EdfReader

def filter_response(band, order):
        fs = 500
        ord = order
        brainwaves = {
            'alpha': [8, 12],
            'beta' : [12, 35],
            'theta': [4, 8],
            'delta': [35, 45],
            'gamma': [0.5,4]
        }

        brainwave = brainwaves[band]
        low_cut = brainwave[0]
        high_cut = brainwave[1]
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = signal.butter(ord, [low, high], btype='bandpass')
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(100, color='green')  # cutoff frequency
        plt.show()
        return

def psd_plot(sub, band, channel, frequency_bins=128):
    str_directory = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\raw_physio_data"

    brainwaves = {
        'alpha': 0,
        'beta': 1,
        'theta': 2,
        'delta': 3,
        'gamma': 4
    }

    brainwave = brainwaves[band]
    person = signal_properties(EdfReader(str_directory + sub))
    brain_wave_matrix = person.brain_wave_bands
    sig = brain_wave_matrix[channel, brainwave , :]
    freqs, psd = signal.welch(sig, frequency_bins, return_onesided=True)
    plt.figure(figsize=(5, 4))
    plt.semilogx(freqs, abs(psd))
    plt.title('PSD: power spectral density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.tight_layout()
    plt.show()
    return

