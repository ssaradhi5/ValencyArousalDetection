import os
import numpy as np

from processing.build_features import signal_properties
from processing.dataset_preparation import feature_preparation
from pyedflib import EdfReader

b_directory = os.fsencode(r"C:\Users\srika\Desktop\URA\Arithmetic\raw_physio_data")
str_directory = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\raw_physio_data"

# Go through list of all subjects and calculate features (power spectral densities, etc.) and then create new dataset
def transform_data(all_tests):
    # 30 rows x 2 = 60 rows
    # 21 channels x 3 brain bands = 63 columns
    combined = np.full([60, 63], None)
    column = [None]*64
    for subject in all_tests:
        # For each subject, calculate features for inactive and active data
        subject_inactive = signal_properties(subject[0])
        subject_active = signal_properties(subject[1])
        # Combine inactive and active data into one list and gather features and labels
        combine_features = list(feature_preparation(subject_inactive, subject_active).reshape_data())
        subject_features = combine_features[0]
        subject_labels = np.expand_dims(combine_features[1], axis=1)
        # Add features and labels of each subject to a table of previous subjects
        row = np.hstack([subject_features, subject_labels])
        column = np.vstack([column, row])

    # save csv of feature extracted dataset
    column = np.delete(column, (0), axis=0)
    path = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\data\processed_datasets"
    np.savetxt(path+r"\feature_label_extra2.csv", np.c_[column], fmt='%s', delimiter=',')
    return combined

# read EDF files and concatenate inactive and active data
def read_all():
    tests_inactive, tests_active = []*36, []*36

    test33_inactive = EdfReader(str_directory + r"\Subject33_1.edf")
    test34_inactive = EdfReader(str_directory + r"\Subject34_1.edf")
    test35_inactive = EdfReader(str_directory + r"\Subject35_1.edf")

    test30_active = EdfReader(str_directory + r"\Subject30_2.edf")
    test31_active = EdfReader(str_directory + r"\Subject31_2.edf")
    test32_active = EdfReader(str_directory + r"\Subject32_2.edf")
    test33_active = EdfReader(str_directory + r"\Subject33_2.edf")
    test34_active = EdfReader(str_directory + r"\Subject34_2.edf")
    test35_active = EdfReader(str_directory + r"\Subject35_2.edf")

    counter_1 = 0
    counter_2 = 0

    for filename in os.listdir(str_directory):
        try:
            if filename.endswith("_1.edf"):
                counter = "{:02d}".format(counter_1)
                name = str_directory + r"\Subject" + fr'{counter}' + r"_1.edf"
                tests_inactive.append(EdfReader(name))
                counter_1 += 1
                continue
            elif filename.endswith("_2.edf"):
                counter = "{:02d}".format(counter_2)
                name = str_directory + r"\Subject" + fr'{counter}' + r"_2.edf"
                tests_active.append(EdfReader(name))
                counter_2 += 1
                continue
            else:
                continue
        except:
            pass

    tests_active.extend(( test30_active, test31_active, test32_active, test33_active, test34_active, test35_active))
    tests_inactive.extend((test33_inactive, test34_inactive, test35_inactive))

    combined_data = tuple(zip(tests_inactive[:30], tests_active[:30]))
    return combined_data