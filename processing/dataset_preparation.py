import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class feature_preparation:
    def __init__(self, inactive, active):
        self.inactive_features = inactive.max_psd
        self.active_features = active.max_psd
        self.inactive_labels = inactive.label
        self.active_labels = active.label
        self.features, self.labels = self.reshape_data()

    def find_feature_map(self, band):
        inactive = self.Inactive
        active = self.Active
        ## graph shows what psd for brain wave for each channel
        ## x axis is channel
        ## y axis is value

        plt.close()
        ## alpha
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = np.arange(0,21,1)
        y1 = [0]*21
        y2 = [0]*21
        for i in range(21):
            y1[i] = inactive[i,band]
            y2[i] = active[i,band]

        ax1.scatter(x,y1, color='r', label='inactive', marker="v")
        ax1.scatter(x,y2, color='g', label='active', marker = "^")
        plt.legend(loc='upper left')
        plt.xlabel('Channels')
        plt.ylabel('Max PSD')
        plt.show()

    def reshape_data(self):
        inactive_feature = (self.inactive_features).flatten()
        active_feature = (self.active_features).flatten()
        inactive_label = (self.inactive_labels).flatten()
        active_label = (self.active_labels).flatten()

        # features = list(zip(inactive_feature,active_feature))
        features = (np.stack((inactive_feature, active_feature)))
        labels = (np.stack((inactive_label, active_label)))
        labels = np.transpose(labels[:,0])
        return features, labels

def construct_dataframe(dataset):
     features = dataset[:, 0:63]
     labels = dataset[:, 63]
     num_channels = 21
     num_bands = 3
     num_samples = 6
     num_sub = 30
     name_array = []
     row_array = []

     this_dict = {
         0: 'alpha',
         1: 'beta',
         2: 'theta'
     }

     for channel in range(num_channels):
         for bands in range(num_bands):
             name_array.append('ch' + str(channel + 1) + '_' + this_dict[bands])

     for sub in range(num_sub):
         for sample in range(num_samples):
             row_array.append('subject_' + str(sub + 1) + '_' + 'sample_' + str(sample + 1))

     data = pd.DataFrame(features, columns=name_array, index=row_array+row_array)
     data.insert(63, "Labels", labels)
     return data

