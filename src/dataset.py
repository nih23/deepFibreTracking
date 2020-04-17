'''
This class is responsible for data loading
'''
import torch
from torch.utils import data

def reverse_line(curr_dwi, curr_next_direction):
    """Calculates an inverse streamline for more efficient training. Returns tuple (dwi, nextDirection).

    Arguments:
    curr_dwi: the DWI information
    curr_next_direction: the matching directions
    """
    # Backwards motion
    curr_next_direction_back = (torch.cat((curr_next_direction[-1:],\
         curr_next_direction[:-1])) * -1).flip([0])
    # backward tracking - circular shift by one, multiply with  -1 and flip on 0 dimension
    curr_dwi_back = curr_dwi.flip([0]) # just flip for input data
    curr_next_direction_back[-1, :] = -1*curr_next_direction_back[-2, :]
    return curr_dwi_back, curr_next_direction_back


class Dataset(data.Dataset):
    """Represents a dataset for training and extends the default torch.utils.data.Dataset . Handles the mirroring of streamlines to artificially increase number. """
    def __init__(self, path="data_ijk.pt"):
        """Initializes the Dataset

        Arguments:
        path: the path to the data file
        """
        self.path = path
        self.max_length = 201
        (self.data_array, self.dir_array, self.len_array) = torch.load(self.path)
        self.len_data = len(self.len_array)
        #self.data_array = self.data_array.cuda()
        #self.dir_array = self.dir_array.cuda()
        #self.len_array = self.len_array.cuda()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.len_data * 2 
    def get_max_length(self):
        """Returns max length of one single sample to prevent unneccessary padding"""
        return self.max_length
    def __getitem__(self, index):
        """Returns the item with given index from dataset
        
        Arguments:
        index: the index
        """
        reverse = index >= len(self)//2
        if reverse:
            index = index - len(self)//2
        dwi = torch.zeros(self.max_length, 2700, dtype=torch.float32, device="cuda")
        next_direction = torch.zeros(self.max_length, 3, dtype=torch.float32, device="cuda")
        slen = self.len_array[index].item()
        dwi_data = self.data_array[index, 0:int(slen), :]
        next_direction_data = self.dir_array[index, 0:int(slen), :]
        if reverse:
            dwi_data, next_direction_data = reverse_line(dwi_data, next_direction_data)
        dwi[:int(slen), :] = dwi_data
        next_direction[:int(slen), :] = next_direction_data
        probability_continue_tracking = \
            (torch.arange(self.max_length) < slen-1).unsqueeze(1).float().cuda()
        return dwi, next_direction, probability_continue_tracking, slen
