"A simple training routine, just testing existing code."
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as dataL

from src.models import ModelLSTM
from src.data import ISMRMDataContainer
from src.data.postprocessing import res100
from src.dataset import StreamlineDataset
from src.tracker import ISMRMReferenceStreamlinesTracker

def collate_fn(el):
    '''
    Padds batch of variable length
    '''
    inputs = [torch.flatten(t[0], start_dim=1) for t in el]
    outputs = [torch.flatten(t[1], start_dim=1) for t in el]

    lengths = torch.tensor([t[0].shape[0] for t in el])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs).double()
    outputs = torch.nn.utils.rnn.pad_sequence(outputs).double()
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

def radians_loss(input_data, target, mask):
    """Quick implementation of the radian loss 1- cos(alpha).

    Arguments:
    input_data: the network output
    target: the supposed output
    mask: mask for masking out unused padding areas. Essential to prevent division by zero."""
    cossim = torch.nn.CosineSimilarity(dim=2)
    output = cossim(input_data, target)**2
    output = output[mask.squeeze() != 0]
    return 1 - torch.mean(output)
def feed_model(model, generator, optimizer=None):
    complete_loss = 0
    progress = 0
    for dwi, next_dir, lengths in generator:
        #print("SHAPES:")
        #print(dwi.shape) # torch.Size([165, 64, 6300])
        #print(next_dir.shape) # torch.Size([165, 64, 3])
        #print(lengths.shape) # torch.Size([64])
        if optimizer is not None:
            optimizer.zero_grad()
        model.reset()
        dwi = dwi.cuda()
        pred_next_dir = model(dwi)
        next_dir = next_dir.cuda()
        mask = (torch.arange(dwi.shape[0])[None, :] < lengths[:, None]).transpose(0, 1).cuda()
        # torch.Size[165, 64]
        #print(pred_next_dir.shape)
        #print(mask.shape)
        pred_next_dir = pred_next_dir * mask[..., None]
        loss = radians_loss(next_dir, pred_next_dir, mask)
        complete_loss = complete_loss + (len(lengths)/len(generator.dataset)) * loss.item()
        progress = progress + len(lengths)
        print("Element {}/{} - loss: {:6.5f}".format(progress, len(generator.dataset), loss),
              end='\r')
        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return complete_loss
def main():
    data = ISMRMDataContainer()
    tracker = ISMRMReferenceStreamlinesTracker(data)
    tracker.track()
    print("Loaded Data")
    dataset = StreamlineDataset(tracker, data, rotate=True, grid_dimension=(7, 3, 3),
                                append_reverse=True, postprocessing=res100())
    training_set, validation_set = getdatasets(dataset)
    print(len(dataset))
    print(len(training_set))
    print("Initialized Dataset")
    sizes = dataset.get_feature_shapes()
    sizes = (torch.prod(torch.tensor(sizes[0])).item()*-1,
             torch.prod(torch.tensor(sizes[1])).item()*-1)

    model = ModelLSTM(dropout=0.05, hidden_sizes=[256, 256], sizes=sizes,
                      activation_function=nn.Tanh()).double().cuda()
    print("Initialized Model")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    params = {'batch_size': 64, 'num_workers': 24, 'shuffle': True, 'collate_fn': collate_fn}
    training_generator = dataL.DataLoader(training_set, **params)
    validation_generator = dataL.DataLoader(validation_set, **params)
    print("Starting to train")
    epochs = 1000
    best_loss = 1e10
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = feed_model(model, training_generator, optimizer=optimizer)
        model.eval()
        with torch.no_grad():
            loss = feed_model(model, validation_generator)
            if loss < best_loss:
                torch.save(model.state_dict(), 'model.pt')
                best_loss = loss
        print(("Epoch {:5d}/{:<5d} - train loss: {:6.5f} - best loss: {:6.5f} - loss: {:6.5f}")
              .format(epoch, epochs, train_loss, best_loss, loss))

if __name__ == "__main__":
    main()
