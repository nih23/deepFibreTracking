"""Simple MLP example training"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as dataL

from dfibert.data import HCPDataContainer
from dfibert.data.postprocessing import res100
from dfibert.dataset.processing import RegressionProcessing
from dfibert.dataset import SingleDirectionsDataset # TODO update example
from dfibert.tracker import CSDTracker
from dfibert.util import random_split

# This is a simple MLP training to show example usage for the provided library



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# switch to cuda if available
# this only works if you want to train on a single CPU/GPU, without parallelization 

class ModelMLP(nn.Module): # the NN as PyTorch module, this is best practise to keep the structure modular and simple. 
    'MLP Model Class'      # but you can also just use an nn.Sequential(*layers) for simple models
    def __init__(self, hidden_sizes=None, activation_function=None, dropout=0, input_size=None):
        """Initializes the MLP model with given parameters:
        
        Arguments:
        hidden_sizes: a list containing the hidden sizes
        activation_function: any torch.nn.* activation function
        dropout: specifies the dropout applied after each layer, if zero, no dropout is applied. default: 0
        input_size: the input size of dwi data.
        """
        super(ModelMLP, self).__init__()

        # construct layers
        hidden_sizes.insert(0, input_size) # input size
        layers = [nn.Flatten(start_dim=1)]
        for index in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[index], hidden_sizes[index+1]))
            layers.append(activation_function)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_sizes[-1], 3))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x): # default function for propagating data through the network, do not change name or arguments. called with mlp_model(data) 
        """pass data through the network. returns output tuple.

        Arguments:
        x: the data to pass 
        """
        return self.main(x)

    
    def compile_model(self, optimizer, loss): # to keep training internal, let optimizer and loss be set external
        self.optimizer = optimizer
        self.loss = loss
    def train_model(self, training_set, validation_set=None, epochs=200):
        "Trains the model"
        best_loss = 10000
        for epoch in range(0, epochs): # each epoch

            self.train() # change model to train mode to activate dropout and backpropagation
            loss = self._feed_model(training_set) 
            print("Epoch {} - train: {:.6f}".format(epoch, loss), end="\r")

            if validation_set is not None:
                self.eval() # change model to evalulation mode to deactivate dropout and backpropagation
                validation_loss = self._feed_model(validation_set, validation=True)

                if validation_loss < best_loss: # save best model to file
                    best_loss = validation_loss
                    torch.save(self.state_dict(), 'best_model.pt')

                print("Epoch {} - train: {:.6f} - test: {:.6f}".format(epoch, loss, validation_loss))

    def _feed_model(self, generator, validation=False): # prepares data in mini batches and passes them through the network
        """Feeds given data to model, returns average loss"""
        whole_loss = 0
        divisor = 0
        for batch, (dwi, next_dir) in enumerate(generator):
            dwi = dwi.to(device)
            next_dir = next_dir.to(device)
            
            print("Batch {}/{}".format(batch, len(generator)), end="\r")

            if not validation:
                self.optimizer.zero_grad()

            pred_next_dir = self(dwi)

            loss = self.loss(pred_next_dir, next_dir)

            if not validation: # backpropagation
                loss.backward()
                self.optimizer.step()

            whole_loss += next_dir.shape[0]*loss.item() # necessary for loss calculation, loss.item() is mean over batch
            divisor += next_dir.shape[0]
            
            # delete tensors to prevent ram usage until generation of next batch
            del dwi
            del next_dir
            del pred_next_dir

        return whole_loss/divisor

def radians_loss(x, y):
    """Quick implementation of the radian loss 1- cos(alpha).

    Arguments:
    x: the network output
    y: the supposed output
    """
    mask = ((y == 0).sum(-1) < 3) # zero vectors in supposed output
    cossim = torch.nn.CosineSimilarity(dim=1)
    output = cossim(x, y)**2
    output = output[mask.squeeze() != 0]
    return 1 - torch.mean(output)

def main():
    "The main function"
    data = HCPDataContainer(100307) # Initialize DW-MRT Image of HCP Participant #100307

    print("Initialized data...")
    tracker = CSDTracker(data, random_seeds=True, seeds_count=10000) # Initialize CSD Tracker to track some streamlines. 
    # In production, you would probably generate the streamlines in a separate file and then prepare them and load them with another Tracker
    tracker.track() # Track the streamlines. This may take some time. It is able to cache the streamlines into cache folder for further executions
    # ! Track the streamlines a head of the data normalization / cropping if you use CSD or DTI Tracker because it won't work with normalized data.
    data.normalize() # normalizes the MRI-data
    data.crop() # crops the MRI image
    print("Initialized streamlines...")

    processing = RegressionProcessing(rotate=False, grid_dimension=(3, 3, 3), postprocessing=res100()) # choose a data Processing option for your training
    dataset = SingleDirectionsDataset(tracker, data, processing, append_reverse=True, online_caching=True) 
    # choose a dataset, this one is good for non-recurrent architectures


    training_set, validation_set = random_split(dataset) # randomly splits the dataset into training and validation. Default 90% Training

    model = ModelMLP(hidden_sizes=[512,512,512], activation_function=nn.ReLU(), dropout=0.05, input_size=2700).to(device) # Initialize the model


    params = {'batch_size': 2048, 'num_workers': 0, 'shuffle': True} # !!!! NUM WORKERS > 0 does not work with caching yet !!!
    training_generator = dataL.DataLoader(training_set, **params) # specify a training and testing generator
    validation_generator = dataL.DataLoader(validation_set, **params)

    print("Initialized dataset & model...")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.compile_model(optimizer, radians_loss) # choose optimizer and loss
    model.train_model(training_generator, validation_set=validation_generator, epochs=5000) # start training

if __name__ == "__main__":
    main()
