import numpy as np

from models import knn_model, rf_model
from processing.dataset_preparation import construct_dataframe
from processing.extract_edf import read_all, transform_data
#  dont dlete above

from visualization.sig_property_visuals import filter_response, psd_plot, knn_visual, plot_correlation



# sys.path.append(r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\DataAttributes\ElectrodeAttributes.py")


""" The data files with EEG are provided in EDF (European Data Format) format. Each folder contains two recording files per subject:
with “_1” postfix -- the recording of the background EEG of a subject (before mental arithmetic task)
with “_2” postfix -- the recording of EEG during the mental arithmetic task.
In this experiment all subjects are divided into two groups:
Group “G” (24 subjects) performing good quality count (Mean number of operations per 4 minutes = 21, SD = 7.4 ),
Group ”B” (12 subjects) performing bad quality count (Mean number of operations per 4 minutes = 7, SD = 3.6).
In the file “subjects_info.xlsx” the “Count quality” column provides info which subjects correspond
to which group (0 - Group ”B”, 1 - Group “G”). Additionally, file “subjects_info.xlsx” provides basic information
about each subject (gender, age, job, date of recording). """

# n = test1_inactive.signals_in_file
# signal_labels = test1_inactive.getSignalLabels()
# sigbufs = np.zeros((n, test1_inactive.getNSamples()[0]))
# fig = plt.figure()
# ax = plt.axes()


# for i in np.arange(n):
#     sigbufs[i, :] = test1_inactive.readSignal(i)
#     ax.plot(test1_inactive.readSignal(i))
#     plt.show()

''' Load raw dataset, process and extract features into a dataframe '''
# raw_dataset = read_all()
# process_dataset = transform_data(raw_dataset)
processed_dataset = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\processed_datasets\TimeFeature.csv"
my_data = construct_dataframe(np.genfromtxt(processed_dataset, delimiter=',').astype('uint32'))


''' data visualization'''

# filter_response(band='theta', order=5)
# psd_plot(r"\Subject33_1.edf", band='alpha', channel=1, frequency_bins=128)
# knn_visual(my_data)
plot_correlation(my_data)
''' Running different ML models'''

knn_model(my_data)
rf_model(my_data)

