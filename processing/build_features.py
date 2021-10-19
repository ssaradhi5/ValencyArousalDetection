import numpy as np
import math

from scipy.signal import butter, lfilter, welch

# Build signal properties class for feature building of brain bands, power spectral densities, time-domain
class signal_properties:
    def __init__(self, patient):
        self.patient = patient
        self.gender = patient.gender
        self.fs = 500
        self.num_channels = patient.signals_in_file
        self.data_length = patient.getNSamples().max()
        self.raw_data_matrix = self.get_data(patient)
        self.brain_wave_bands = self.calc_brain_bands(self.raw_data_matrix)
        self.psd = self.calc_psd(self.brain_wave_bands)
        self.max_psd = self.calc_peak_psd()
        self.label = self.attach_label(patient)

    # Access object and read the signal to get time series
    def get_data(self, signals):
        all_signals = np.full([self.num_channels, self.data_length], None)

        for channel in range(self.num_channels):
            all_signals[channel] = signals.readSignal(channel)

        # windowing for splitting up time series into segments
        min = 1 * (self.data_length // 3)
        max = 2 * (self.data_length // 3)
        all_signals = all_signals[:, min:max]
        return all_signals

    # Add label to the data matrix depending on if the file name has _1 or _2
    def attach_label(self, patient):

        name = patient.file_name
        label_inactive = "_1"
        label_active = "_2"

        if label_inactive in name:
            label = np.full([self.num_channels, 3], 0)
            return label
        if label_active in name:
            label = np.full([self.num_channels, 3], 1)
            return label

    # calculate max power spectral density value for each brain wave, channel, subject
    def calc_peak_psd(self):
        psd = self.psd
        peak_psd = np.full([self.num_channels, 3], 0)

        for ch in range(self.num_channels):
            for band in range(3):
                peak_psd[ch, band] = np.amax(psd[ch, band, :])

        return peak_psd

    # use welch method to get power density spectrum and save the magnitude
    def calc_psd(self, brain_signals):
        fs = self.fs
        frequency_bins = 256

        # math.ceil(frequency_bins / 2) + 1

        # returns based on half the frequency bins so make length array = freqbin/2
        # 5 in the argument is for num bands
        pxx_den = np.full([self.num_channels, 3, math.ceil(frequency_bins / 2) + 1], 0)

        # iterate through each channel and calculate welch method for each brain wave band
        for channel in range(self.num_channels):

            f_alpha, pxx_alpha= welch(brain_signals[channel, 0, :], frequency_bins, return_onesided=True)
            f_beta, pxx_beta = welch(brain_signals[channel, 1, :], frequency_bins, return_onesided=True)
            f_theta, pxx_theta = welch(brain_signals[channel, 2, :], frequency_bins, return_onesided=True)
            # f_gamma, pxx_gamma = welch(brain_signals[channel, 3, :], frequency_bins, return_onesided=True)
            # f_delta, pxx_delta  = welch(brain_signals[channel, 3, :], frequency_bins, return_onesided=True)

            pxx_den[channel,0,:] = abs(pxx_alpha)
            pxx_den[channel,1,:] = abs(pxx_beta)
            pxx_den[channel,2,:] = abs(pxx_theta)
            # pxx_den[channel,3,:] = abs(pxx_gamma)
            # pxx_den[channel,4,:] = abs(pxx_delta)

        return pxx_den

    # split the time series data into different brain bands while butterworth filtering
    def calc_brain_bands(self, all_channels):
        Alpha = [8, 12]
        Beta = [12, 35]
        Theta = [4, 8]
        # Gamma = [35, 45]
        # Delta = [0.5, 4]
        num_channels = self.num_channels
        data_length = self.data_length // 3
        num_brain_bands = 3
        fs = 500
        brain_wave_matrix = np.full([num_channels, num_brain_bands, data_length], None)

        # butterworth filter to get the smoothest roll-off
        for channel in range(num_channels):
            signal = all_channels[channel, :]
            brain_wave_matrix[channel, 0, :] = butter_bandpass_filter(signal, Alpha, fs, order=5)
            brain_wave_matrix[channel, 1, :] = butter_bandpass_filter(signal, Beta, fs, order=5)
            brain_wave_matrix[channel, 2, :] = butter_bandpass_filter(signal, Theta, fs, order=5)
            # brain_wave_matrix[channel, 3, :] = butter_bandpass_filter(signal, Gamma, fs, 'Gamma', order=5)
            # brain_wave_matrix[channel, 4, :] = butter_bandpass_filter(signal, Delta, fs, 'Delta', order=5)
        return brain_wave_matrix

# butterworth bandpass filtering with order 5
def butter_bandpass_filter(data, band, fs, order=5):
    low_cut = band[0]
    high_cut = band[1]
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y