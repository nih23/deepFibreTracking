'''
Class representing the models and model generation
'''
import random
import types

import torch
import torch.nn as nn


def get_random_model(network_type=None, dataset_type=None, batch_size=None, layer_size=None,\
    depth=None, learning_rate=None, dropout=None, activation_function=None):
    """Returns a model with given parameters. Every parameter which equals None will be replaced by a random value in search space.
    
    Parameters:
    network_type: sets network type. values:['mlp','lstm']. default: None
    dataset_type: sets dataset type. values:['1x','3x']. default: None
    batch_size: specifies batch size. default: None
    layer_size: sets layer_size. default: None
    depth: sets network depth. default: None
    learning_rate: specifies learning rate. default: None
    dropout: specifies dropout behind each layer. default: None
    activation_function: set activation_function to use. values: every torch.nn activation_function. default: None
    """
    if network_type is None:
        network_type = random.choice(["mlp", "lstm"])
    if dataset_type is None:
        dataset_type = random.choice(["1x", "3x"])
    if batch_size is None:
        batch_size = int(2**(random.randint(400, 1100)/100))
    if layer_size is None:
        layer_size = int(2**(random.randint(400, 900)/100))
    if depth is None:
        depth = random.randint(2, 6)
    if learning_rate is None:
        learning_rate = 10**(random.randint(-600, -300)/100)
    if dropout is None:
        dropout = random.randint(0, 50)/100
    if activation_function is None:
        activation_function = random.choice([nn.Tanh(), nn.ReLU(), nn.LeakyReLU()])

    state = types.SimpleNamespace()
    state.network_type = network_type
    state.dataset_type = dataset_type
    state.batch_size = batch_size
    state.layer_size = layer_size
    state.depth = depth
    state.learning_rate = learning_rate
    state.dropout = dropout
    state.activation_function = activation_function
    state.crop = dataset_type == "1x"
    state.no_epochs = 200
    input_size = 2700
    if state.crop:
        input_size = 100
    layers = [layer_size] * depth

    if network_type == "mlp":
        model = ModelMLP(dropout=dropout, hidden_sizes=layers, input_size=input_size,\
             activation_function=activation_function)
    else:
        model = ModelLSTM(dropout=dropout, hidden_sizes=layers, input_size=input_size,\
             activation_function=activation_function)
    model = model.cuda()

    return (model, state)



class ModelMLP(nn.Module):
    'MLP Model Class'
    def __init__(self, hidden_sizes=None, activation_function=None, dropout=0, input_size=None):
        """Initializes the MLP model with given parameters:
        
        Arguments:
        hidden_sizes: a list containing the hidden sizes
        activation_function: any torch.nn.* activation function
        dropout: specifies the dropout applied after each layer, if zero, no dropout is applied. default: 0
        input_size: the input size of dwi data.
        """
        super(ModelMLP, self).__init__()
        hidden_sizes.insert(0, input_size) # input size
        self.hidden = nn.ModuleList()
        self.activation_function = activation_function
        for index in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[index], hidden_sizes[index+1]))
            if dropout > 0:
                self.hidden.append(nn.Dropout(p=dropout))
        self.output1 = nn.Linear(hidden_sizes[-1], 3)
        self.output2 = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        """pass data through the network. returns output tuple.

        Arguments:
        x: the data to pass 
        """
        for layer in self.hidden:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_function(x)
        output_1 = torch.tanh(self.output1(x)) # hidden state & cell state
        output_2 = torch.sigmoid(self.output2(x)) # Layer mit sigmoid
        return (output_1, output_2)

class ModelLSTM(nn.Module):
    'LSTM Model'
    def __init__(self, hidden_sizes=None, activation_function=None, dropout=0, input_size=2700):
         """Initializes the LSTM model with given parameters:
        
        Arguments:
        hidden_sizes: a list containing the hidden sizes
        activation_function: any torch.nn.* activation function
        dropout: specifies the dropout applied after each layer, if zero, no dropout is applied. default: 0
        input_size: the input size of dwi data.
        """
        super(ModelLSTM, self).__init__()
        # [timestep, batch, feature=100]
        self.hidden_state = [None] * len(hidden_sizes)
        self.hidden_len = len(hidden_sizes)
        hidden_sizes.insert(0, input_size) # input size
        self.hidden = nn.ModuleList()
        self.activation_function = activation_function
        for index in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.LSTM(hidden_sizes[index], hidden_sizes[index+1]))
            if dropout > 0:
                self.hidden.append(nn.Dropout(p=dropout))
        self.output1 = nn.Linear(hidden_sizes[-1], 3)
        self.output2 = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        """pass data through the network. returns output tuple.

        Arguments:
        x: the data to pass 
        """
        i = 0
        for layer in self.hidden:
            if isinstance(layer, nn.LSTM):
                x, self.hidden_state[i] = layer(x, self.hidden_state[i])
                x = self.activation_function(x)
                i = i + 1
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
        output_1 = torch.tanh(self.output1(x)) # hidden state & cell state
        output_2 = torch.sigmoid(self.output2(x)) # Layer mit sigmoid
        return (output_1, output_2)
    def reset(self):
        """Resets hidden state of model
        """
        # array [(h, c), ]
        self.hidden_state = [None] * (self.hidden_len)
        # minus 1 because of input size appended in line 44
    def set(self, new_state):
        """Sets hidden state.

        Arguments:
        new_state: the new hidden state.
        """
        self.hidden_state = new_state
