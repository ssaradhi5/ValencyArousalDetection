from pyedflib import EdfReader
from visualization.sig_property_visuals import filter_response, psd_plot, knn_visual
from processing.dataset_preparation import construct_dataframe
import matplotlib.pyplot as plt
import numpy as np
import unittest

class testsanity(unittest.TestCase):

    @classmethod
    def setUpClass(data):
        data.raw_src = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\raw_physio_data\Subject00_1.edf"
        data.raw_sig = EdfReader(data.raw_src)
        data.processed_dataset_src = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\processed_datasets\TimeFeature.csv"
        data.processed_dataset = construct_dataframe(np.genfromtxt(data.processed_dataset_src, delimiter=',').astype('uint32'))

    def test_data_extraction(self):
        test1_inactive = self.raw_sig
        n = test1_inactive.signals_in_file
        sigbufs = np.zeros((n, test1_inactive.getNSamples()[0]))
        ax = plt.axes()

        for i in np.arange(n):
            sigbufs[i, :] = test1_inactive.readSignal(i)
            ax.plot(test1_inactive.readSignal(i))
            plt.show()
        return

    def test_filter_response(self):
        filter_response(band='theta', order=5)
        return

    def test_psd_plot(self):
        psd_plot(r"\Subject33_1.edf", band='alpha', channel=1, frequency_bins=128)
        return

    def test_knn_feature_map(self):
        dataset = self.processed_dataset
        knn_visual(dataset)
        return

if __name__ == '__main__':
    unittest.main()








