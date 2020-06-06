import torch
import torch.nn as nn

class ModelLSTM(nn.Module):
    'LSTM Model'
    def __init__(self, hidden_sizes=None, activation_function=None, dropout=0, sizes=(2700 , 3)):
        """Initializes the LSTM model with given parameters.

        Parameters
        ----------
        hidden_sizes: list
            A list of containing the hidden sizes as integers.
        activation_function: func
            any torch.autograd.Function as activation function.
        dropout: float
            specifies the dropout applied after each layer. By default, no dropout is applied.
        input_size: int
            the input size of dwi data.
        """
        super(ModelLSTM, self).__init__()
        # [timestep, batch, feature=100]
        self.hidden_state = [None] * len(hidden_sizes)
        self.hidden_len = len(hidden_sizes)
        hidden_sizes.insert(0, sizes[0]) # input size
        self.hidden = nn.ModuleList()
        self.activation_function = activation_function
        for index in range(self.hidden_len):
            self.hidden.append(nn.LSTM(hidden_sizes[index], hidden_sizes[index+1]))
            if dropout > 0:
                self.hidden.append(nn.Dropout(p=dropout))
        self.output = nn.Linear(hidden_sizes[-1], sizes[1])

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
        output = torch.tanh(self.output(x)) # hidden state & cell state
        return output
    def reset(self):
        """Resets hidden state of model
        """
        # array [(h, c), ]
        self.hidden_state = [None] * (self.hidden_len)
    def set(self, new_state):
        """Sets hidden state.
        Arguments:
        new_state: the new hidden state.
        """
        self.hidden_state = new_state
