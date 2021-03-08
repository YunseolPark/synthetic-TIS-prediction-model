import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset

class TISDataset(Dataset):
    """
    Class to generate TIS dataset for dataloader
    """

    def read_dna(self, file, class_id):
        """
        Reads DNA files and save to a list of sequences

        Args:
            file: file that contains DNA sequences
            class_id: class of the sequence file (positive: 1, negative: 0)
        Return:
            List that contains tuples consisting of a sequence and class id for the given file
        """
        dna_list = []
        # read and save file as a list with the corresponding class
        for line in open(file):
            dna_list.append((line.strip(), class_id))
        return dna_list

    def __init__(self, pos_data, neg_data):
        # Make a list of DNA with class id
        self.dna_list = self.read_dna(pos_data, 1)          # Positive: 1
        self.dna_list.extend(self.read_dna(neg_data, 0))    # Negative: 0
        self.data_len = len(self.dna_list)

    def __getitem__(self, index):
        # Read data
        dna_data, label = self.dna_list[index]
        dna = {'A': [0], 'G': [1], 'C': [2], 'T': [3], 'N':[0,1,2,3], 'R':[0,1], 'Y':[2,3], 'K':[1,3], 'S':[1,2]}
        # Assign values
        one_hot = np.zeros((len(dna_data), 4))
        for i, nuc in enumerate(dna_data):
        	number = 1 / len(dna[nuc])
        	for position in dna[nuc]:
        		one_hot[i, position] = number
        # Convert numpy to tensor
        tensor_dna = torch.from_numpy(one_hot).float()
        return tensor_dna, label

    def __len__(self):
        return self.data_len