import numpy as np

from models import knn_model, rf_model, k_selector
from processing.dataset_preparation import construct_dataframe, pca_analysis
from processing.extract_edf import read_all, transform_data
from visualization.sig_property_visuals import plot_correlation


""" The data files with EEG are provided in EDF (European Data Format) format. Each folder contains two recording files per subject:
with “_1” postfix -- the recording of the background EEG of a subject (before mental arithmetic task)
with “_2” postfix -- the recording of EEG during the mental arithmetic task.
In this experiment all subjects are divided into two groups:
Group “G” (24 subjects) performing good quality count (Mean number of operations per 4 minutes = 21, SD = 7.4 ),
Group ”B” (12 subjects) performing bad quality count (Mean number of operations per 4 minutes = 7, SD = 3.6).
In the file “subjects_info.xlsx” the “Count quality” column provides info which subjects correspond
to which group (0 - Group ”B”, 1 - Group “G”). Additionally, file “subjects_info.xlsx” provides basic information
about each subject (gender, age, job, date of recording). """

''' Load raw dataset, process and extract features into a dataframe '''
# raw_dataset = read_all()
# process_dataset = transform_data(raw_dataset)
processed_dataset = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\processed_datasets\TimeFeature.csv"
my_data = construct_dataframe(np.genfromtxt(processed_dataset, delimiter=',').astype('uint32'))

'''Feature Analysis'''
my_data.describe()
pca_analysis(my_data)
k_selector(my_data)
plot_correlation(my_data)

''' Running different ML models'''
knn_model(my_data)
rf_model(my_data)
