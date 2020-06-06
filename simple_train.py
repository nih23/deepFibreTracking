"A simple training routine, just testing existing code."
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as dataL

from src.models import ModelLSTM
from src.data import ISMRMDataContainer
from src.data.postprocessing import res100
from src.dataset import StreamlineDataset, StreamlineClassificationDataset
from src.tracker import CSDTracker, ISMRMReferenceStreamlinesTracker

def collate_fn(inputs, outputs):
    '''
    Padds batch of variable length
    '''
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in inputs]).cuda()
    ## padding and dimension swap
    inputs = torch.nn.utils.rnn.pad_sequence(inputs).cuda()
    outputs = torch.nn.utils.rnn.pad_sequence(outputs).cuda()
    return inputs, outputs, lengths

def getdatasets(dataset, training_part=0.9):
    """Retrieves a dataset from given path and splits them randomly in train and test data.
        Arguments:
        dataset: the dataset
        training_part: float indicating the percentage of training data. default: 0.9 (90%)
    """
    train_len = int(training_part*len(dataset))
    test_len = len(dataset) - train_len
    (train_split, test_split) = torch.utils.data.random_split(dataset, (train_len, test_len))
    return train_split, test_split
def feed_model(model, generator, optimizer=None):
    mse_loss = torch.nn.MSELoss()
    for dwi, next_dir, lengths in generator:
        print("SHAPES:")
        print(dwi.shape)
        print(next_dir.shape)
        print(lengths.shape)
        if optimizer is not None:
            optimizer.zero_grad()
        model.reset()
        batch_len = len(dwi)
        mask =  torch.arange(200)[None, :] < lengths[:, None]
    return 0
def main():
    data = ISMRMDataContainer(denoise=True)
    tracker = ISMRMReferenceStreamlinesTracker()
    print("Loaded Data")
    dataset = StreamlineDataset(tracker, data, rotate=True, grid_dimension=(7, 3, 3),
                                append_reverse=True, postprocessing=res100())
    training_set, validation_set = getdatasets(dataset)
    print("Initialized Dataset")
    sizes = dataset.get_feature_shapes()
    sizes = (torch.prod(sizes[0]), torch.prod(sizes[1]))

    model = ModelLSTM(dropout=0.05, hidden_sizes=[256, 256], sizes=sizes,
                      activation_function=nn.Tanh()).cuda()
    print("Initialized Model")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    params = {'batch_size': 64, 'shuffle': True, 'collate_fn': collate_fn}
    training_generator = dataL.DataLoader(training_set, **params)
    validation_generator = dataL.DataLoader(validation_set, **params)
    print("Starting to train")
    epochs = 1000
    for epoch in range(1, epochs + 1):
        model.train()
        _ = feed_model(model, training_generator, optimizer=optimizer)
        model.eval()
        with torch.no_grad():
            pass
            #(loss, rad_loss, pStop_loss) = feed_model(model, validation_generator)
        print(("Epoch {:5d}/{:<5d} - loss: {:6.5f} rad_loss :"
               "{:6.5f} pStop_loss: {:6.5f} ").format(epoch, epochs, loss, rad_loss, pStop_loss))
if __name__ == "__main__":
   main()

