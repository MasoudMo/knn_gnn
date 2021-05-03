"""
Author: Masoud Mokhtari

Data preprocessing including filtering, Fast Fourier Transform (FFT) and dimensionality reduction on ECG signals.

"""

import wfdb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import os
import dgl
from sklearn.decomposition import PCA
from datetime import datetime
import logging
import torch
from scipy import fft
from scipy import signal

try:
    from knn_cuda import KNN
except ImportError:
    pass

logger = logging.getLogger('ecg_gnn')


def create_knn_adj_mat(features, k, weighted=False, n_jobs=None, algorithm='auto', threshold=None, use_gpu=False):
    """
    Create a directed normalized adjacency matrix from input nodes based on k-nearest neighbours
    Parameters:
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        weighted (bool): set to True for weighted adjacency matrix (based on Euclidean distance)
        n_jobs (int): number of jobs to deploy if GPU is not used
        algorithm (str): Choose between auto, ball_tree, kd_tree or brute
        threshold (float): Cutoff value for the Euclidean distance
        use_gpu (bool): Indicates whether GPU is to be used for the KNN algorithm
    Returns:
        (coo matrix): adjacency matrix as a sparse coo matrix
    """
    t_start = datetime.now()

    if use_gpu:

        features_extra_dim = np.expand_dims(features, axis=2)

        knn = KNN(k=k, transpose_mode=True)

        # Find the k nearest neighbours and their distance
        dist, idx = knn(torch.from_numpy(features_extra_dim).cuda(),
                        torch.from_numpy(features_extra_dim).clone().cuda())

        torch.cuda.empty_cache()

        del features_extra_dim

        idx = idx.cpu()
        dist = dist.cpu()

        # Clean up the indices and distances
        dist = dist.flatten()

        # Create tuples of indices where an edge exists
        rows = np.repeat(np.arange(features.shape[0]), k)
        columns = idx.flatten()
        non_zero_indices = tuple(np.stack((rows, columns)))

        del rows
        del columns
        del idx

        # Remove edges where the distance is higher than the threshold
        if threshold:
            indices_to_remove = dist > threshold
            indices_to_remove = np.where(indices_to_remove)
            non_zero_indices = tuple(np.delete(non_zero_indices, indices_to_remove, 1))
            dist = np.delete(dist, indices_to_remove[0], 0)

            del indices_to_remove

        if weighted:

            # Create zero matrix as the initial adjacency matrix
            adj_mat_weighted = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)

            # Fill in the adjacency matrix with node distances
            adj_mat_weighted[non_zero_indices] = dist

            non_zero_indices = np.nonzero(adj_mat_weighted)

            # Take reciprocal of non-zero elements to associate lower weight to higher distances
            adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

            # Normalize rows
            coo_matrix = sp.coo_matrix(adj_mat_weighted)
            normalized_coo_matrix = normalize(coo_matrix)

            # DGL requires self loops
            normalized_coo_matrix = normalized_coo_matrix + sp.eye(normalized_coo_matrix.shape[0])

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return normalized_coo_matrix

        else:
            # Create eye matrix as the initial adjacency matrix
            adj_mat_binary = np.zeros((features.shape[0], features.shape[0]))

            # Create the binary adjacency matrix
            adj_mat_binary[non_zero_indices] = 1

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return sp.coo_matrix(adj_mat_binary)
    else:

        # initialize and fit nearest neighbour algorithm
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, algorithm=algorithm)
        neigh.fit(features)

        # Obtain matrix with distance of k-nearest points
        adj_mat_weighted = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='distance')).toarray())

        if threshold:
            indices_to_zero = adj_mat_weighted > threshold
            adj_mat_weighted[indices_to_zero] = 0

        non_zero_indices = np.nonzero(adj_mat_weighted)

        if weighted:
            # Take reciprocal of non-zero elements to associate lower weight to higher distances
            adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

            # Normalize rows
            adj_mat_weighted = sp.coo_matrix(adj_mat_weighted)
            normalized_coo_matrix = normalize(adj_mat_weighted)

            # DGL requires self loops
            normalized_coo_matrix = normalized_coo_matrix + sp.eye(normalized_coo_matrix.shape[0])

            del adj_mat_weighted

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return normalized_coo_matrix

        # Obtain the binary adjacency matrix
        adj_mat_binary = adj_mat_weighted
        adj_mat_binary[non_zero_indices] = 1
        adj_mat_binary = adj_mat_binary + np.eye(adj_mat_binary.shape[0])

        del adj_mat_weighted

        t_end = datetime.now()
        logger.debug("it took {} to create the graph".format(t_end - t_start))

        return sp.coo_matrix(adj_mat_binary)


def collate(samples):
    """
    Collate function used by the data loader to put graphs into a batch
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.float)


class PtbEcgDataSet(Dataset):
    """
    Dataset class for the PTB ECG dataset
    """

    def __init__(self,
                 root_dir,
                 records,
                 weighted,
                 records_to_exclude=None,
                 k=1,
                 n_jobs=1,
                 fft_data=False,
                 filter=False,
                 fft_graph=True,
                 samp_from=0,
                 samp_to=1000,
                 fft_points_to_keep=None,
                 save_plots=False,
                 file_path="./",
                 pca_dim=None
                 ):
        """
        Constructor for the PTB ECG dataset
        Parameters:
            root_dir (str): Path to root data directory
            records (str): Path to file with local directories to patient record files
            weighted (bool): Indicates whether created graph is weighted or not
            records_to_exclude (list of ints): List of indices to exclude from the records
            k (int): Number of neighbours to find for the KNN graph
            n_jobs (int): Number of jobs to deploy for KNN graph creation
            fft_data(bool): Indicates whether time or freq data is used for node features
            fft_graph (bool): Indicates whether FFT is used for graph creation or not
            filter(bool): Indicates whether filtering is performed on FFT data
            samp_from (int): Index where data is taken from samples
            samp_to (int): Index to where the data is taken from samples
            fft_points_to_keep (int): Indicates how many fft points to keep if fft is used for features or graph cration
            save_plots (bool): Indicates whether data plots are created and saved or not
            file_path (str): Indicates the directory to save plots to
            pca_dim (int): Dimension size to reduce signals to
        """

        # These records are not diagnosed
        if records_to_exclude is None:
            records_to_exclude = [421, 348, 536, 338, 358, 429, 395, 377, 419, 398, 367, 412, 416, 522, 333, 523, 378,
                                  375, 397, 519, 530, 406, 524, 355, 356, 407, 417]

        # Dictionary of diagnosis
        self.diagnosis = {'Reason for admission: Healthy control': 0,
                          'Reason for admission: Myocardial infarction': 1,
                          'Reason for admission: Heart failure (NYHA 2)': 2,
                          'Reason for admission: Bundle branch block': 3,
                          'Reason for admission: Dysrhythmia': 4,
                          'Reason for admission: Myocardial hypertrophy': 5,
                          'Reason for admission: Valvular heart disease': 6,
                          'Reason for admission: Myocarditis': 7,
                          'Reason for admission: Hypertrophy': 8,
                          'Reason for admission: Cardiomyopathy': 9,
                          'Reason for admission: Heart failure (NYHA 3)': 10,
                          'Reason for admission: Unstable angina': 11,
                          'Reason for admission: Stable angina': 12,
                          'Reason for admission: Heart failure (NYHA 4)': 13,
                          'Reason for admission: Palpitation': 14}

        # Read the records file
        self.root_dir = root_dir
        records_dir = os.path.join(root_dir, records)
        records_file = open(records_dir)
        self.records_dirs = [os.path.join(root_dir, s) for s in records_file.read().split('\n')]

        # Ignore last entry in RECORDS file as that's just a new line
        self.records_dirs = self.records_dirs[:-1]

        # Remove data points with no diagnosis
        self.records_dirs = [i for j, i in enumerate(self.records_dirs) if j not in records_to_exclude]

        # Extract number of samples
        self.num_samples = len(self.records_dirs)

        # Other parameters
        self.weighted = weighted
        self.k = k
        self.n_jobs = n_jobs
        self.samp_from = samp_from
        self.samp_to = samp_to
        self.fft_data = fft_data
        self.fft_graph = fft_graph
        self.fft_points_to_keep = fft_points_to_keep
        self.pca_dim = pca_dim
        self.filter = filter

    def __getitem__(self, idx):
        """
        Item iterator for the prostate cancer dataset
        Parameters:
            idx (int): index of data point to retrieve
        Returns:
            (dgl graph): DGL graph with nodes containing nodes with ecg data as features
            (tensor(int)): Label indicating whether the core is cancerous or healthy
        """

        record = wfdb.io.rdrecord(self.records_dirs[idx], sampfrom=self.samp_from, sampto=self.samp_to)
        time_domain_data = np.transpose(record.p_signal).astype(np.float)

        if self.filter:
            cut_off_freq = 5 / (record.fs/2)

            # Generate filter coefficients
            b, a = signal.butter(5, cut_off_freq, 'highpass')

            filtered_time_domain_data = np.empty_like(time_domain_data)

            for idx in range(time_domain_data.shape[0]):
                filtered_time_domain_data[idx, :] = signal.filtfilt(b, a, time_domain_data[idx, :])

        if self.fft_graph or self.fft_data:
            if self.fft_points_to_keep:
                if self.filter:
                    # Find FFT and only keep the first half (half of sampling frequency)
                    freq_domain_data = np.abs(fft(filtered_time_domain_data))[:, :self.fft_points_to_keep].astype(np.float)
                else:
                    freq_domain_data = np.abs(fft(time_domain_data))[:, :self.fft_points_to_keep].astype(np.float)
            else:
                if self.filter:
                    # Find FFT and only keep the first half (half of sampling frequency)
                    freq_domain_data = np.abs(fft(cut_off_freq))[:, :(int(cut_off_freq.shape[1]/2))].astype(np.float)
                else:
                    freq_domain_data = np.abs(fft(time_domain_data))[:, :(int(time_domain_data.shape[1] / 2))].astype(
                        np.float)

        if self.fft_graph:
            adjacency_mat = create_knn_adj_mat(freq_domain_data,
                                               k=self.k,
                                               weighted=self.weighted,
                                               n_jobs=self.n_jobs,
                                               algorithm='auto')
        else:
            adjacency_mat = create_knn_adj_mat(time_domain_data,
                                               k=self.k,
                                               weighted=self.weighted,
                                               n_jobs=self.n_jobs,
                                               algorithm='auto')

        g = dgl.from_scipy(adjacency_mat)

        if self.fft_data:
            if self.pca_dim:
                pca = PCA(n_components=self.pca_dim)
                pca.fit(freq_domain_data)
                g.ndata['x'] = torch.from_numpy(pca.transform(freq_domain_data))
            else:
                g.ndata['x'] = torch.from_numpy(freq_domain_data)
        else:
            if self.pca_dim:
                if self.filter:
                    pca = PCA(n_components=self.pca_dim)
                    pca.fit(filtered_time_domain_data)
                    g.ndata['x'] = torch.from_numpy(pca.transform(filtered_time_domain_data))
                else:
                    pca = PCA(n_components=self.pca_dim)
                    pca.fit(time_domain_data)
                    g.ndata['x'] = torch.from_numpy(pca.transform(time_domain_data))
            else:
                if self.filter:
                    g.ndata['x'] = torch.from_numpy(filtered_time_domain_data)
                else:
                    g.ndata['x'] = torch.from_numpy(time_domain_data)

        # Obtain the label for the specified record
        label = self.diagnosis[record.comments[4]]

        # Only 3 categories (Mycordial infarction, healthy, other diseases)
        if (label is not 0) and (label is not 1):
            label = 1

        return g, label

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cores
        """
        return self.num_samples
