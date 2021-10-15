import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from processing.build_features import signal_properties
from pyedflib import EdfReader
import seaborn as sns

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

def plot_correlation(dataset):

    # Triangle Correlation of Independent Variables with the Dependent Variable
    mask = np.triu(np.ones_like(dataset.corr(), dtype=np.bool))
    figure1 = plt.figure()
    correlation = sns.heatmap(dataset.corr(), annot=False, vmax = 1, vmin = -1, mask=mask,cmap='BrBG')
    print(correlation)
    plt.show()
    figure1.savefig(r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\results\correlation_heatmap.png")
    figure2 = plt.figure()
    # Correlation of Independent Variables with the Dependent Variable
    correlate_ind_dep = sns.heatmap(dataset.corr()[['Labels']].sort_values(by='Labels'), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    print(correlate_ind_dep)
    plt.show()
    figure2.savefig(r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\results\correlation_ind_dep.png")
    return

def knn_visual(dataset):
    features, labels = dataset.iloc[:,0:63].values, dataset.iloc[:,63].values
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors

    n_neighbors = 3
    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = features[:,:2]
    y = labels

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = ['darkorange', 'darkblue']

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(64, 64))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y,
                        palette=cmap_bold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))
        plt.xlabel('slat')
        plt.ylabel('slime')

    plt.show()

    return




