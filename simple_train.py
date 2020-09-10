"A simple training routine, just testing existing code."

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as dataL

from src.models import ModelLSTM
from src.data import ISMRMDataContainer
from src.data.postprocessing import res100
from src.dataset.processing import RegressionProcessing
from src.dataset import StreamlineDataset
from src.tracker import ISMRMReferenceStreamlinesTracker
from src.util import random_split
from src.callbacks import SimpleLogger

def radians_loss(x, y):
    """Quick implementation of the radian loss 1- cos(alpha).

    Arguments:
    x: the network output
    y: the supposed output
    """
    mask = ((y == 0).sum(-1) < 3) # zero vectors in supposed output
    cossim = torch.nn.CosineSimilarity(dim=2)
    output = cossim(x, y)**2
    output = output[mask.squeeze() != 0]
    return 1 - torch.mean(output)

def main():
    "The main function"
    data = ISMRMDataContainer()
    print("Initialized data...")
    tracker = ISMRMReferenceStreamlinesTracker(data, streamline_count=10000)
    tracker.track()
    print("Initialized streamlines...")

    processing = RegressionProcessing(rotate=False, grid_dimension=(3, 3, 3),
                                      postprocessing=res100())
    dataset = StreamlineDataset(tracker, data, processing, append_reverse=True, ram_caching=True)
    training_set, validation_set = random_split(dataset)

    model = ModelLSTM(dropout=0.05, hidden_sizes=[256, 256], sizes=dataset.get_feature_shapes(),
                      activation_function=nn.Tanh()).cuda()


    params = {'batch_size': 64, 'num_workers': 0, 'shuffle': True, 'collate_fn': model.collate_fn}
    training_generator = dataL.DataLoader(training_set, **params)
    validation_generator = dataL.DataLoader(validation_set, **params)

    print("Initialized dataset & model...")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.compile_model(optimizer, radians_loss, metrics=[])
    model.train_model(training_generator, validation_set=validation_generator, epochs=200,
                      callbacks=[SimpleLogger()])

if __name__ == "__main__":
    main()
