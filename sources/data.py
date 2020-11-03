import wfdb
import numpy as np
import matplotlib.pyplot as plt
import wfdb.plot as wfplt
from torch.utils.data import Dataset
import os


class PtbEcgDataSet(Dataset):
    """
    Dataset class for the PTB ECG dataset
    """

    def __init__(self,
                 root_dir,
                 records,
                 weighted,
                 records_to_exclude=None,
                 ):
        """
        Constructor for the PTB ECG dataset
        Parameters:
            root_dir (str): Path to root data directory
            records (str): Path to file with local directories to patient record files
            weighted (bool): Indicates whether created graph is weighted or not
            records_to_exclude (list of ints): List of indices to exclude from the records
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

        # Remove data points with no diagnosis
        self.records_dirs = [i for j, i in enumerate(self.records_dirs) if j not in records_to_exclude]

        self.num_samples = len(self.records_dirs)

        self.weighted = weighted

    def __getitem__(self, idx):
        """
        Item iterator for the prostate cancer dataset
        Parameters:
            idx (int): index of data point to retrieve
        Returns:
            (numpy array): Numpy array containing the signals for a single core
            (int): Label indicating whether the core is cancerous or healthy
        """

        record = wfdb.io.rdrecord(self.records_dirs[idx], sammpto=31000)

        self.size.append(record.p_signal.shape[0])

        # Obtain the label for the specified record
        label = self.diagnosis[record.comments[4]]

        return label

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cores
        """
        return self.num_samples
