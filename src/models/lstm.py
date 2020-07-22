"The LSTM Model"
from math import ceil
import torch
import torch.nn as nn

from src.config import Config

class ModelLSTM(nn.Module):
    'LSTM Model'
    def __init__(self, hidden_sizes=None, activation_function=None, dropout=0, sizes=(2700, 3)):
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
        self.optimizer = None
        self.loss = None
        self.stop_training = False
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

    def compile_model(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def train_model(self, training_set, validation_set=None, epochs=None, callbacks=[]):
        "Trains the model"
        if self.optimizer is None or self.loss is None:
            return #TODO - Add Error if model is not compiled 

        self.stop_training = False
        self.callbacks = callbacks

        callback_params = {
            'batch_size': training_set.batch_size,
            'epochs': epochs,
            'steps': ceil(len(training_set.dataset)/training_set.batch_size),
            'samples': len(training_set.dataset),
            'do_validation': validation_set is not None,
            'metrics': self.metrics,
        }
        if validation_set is not None:
            callback_params['val_batch_size']= validation_set.batch_size
            callback_params['val_steps'] = ceil(len(validation_set.dataset)/
                                                validation_set.batch_size)
            callback_params['val_samples'] = len(validation_set.dataset)

        config = Config.get_config()
        if epochs is None:
            epochs = config.getint("TrainingOptions", "epochs",
                                   fallback="200")


        for callback in self.callbacks:
            callback.set_model(self)
            callback.set_params(callback_params)
            callback.on_train_begin(logs={})

        for epoch in range(0, epochs):

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs={})

            self.train()
            train_metrics = self._feed_model(training_set)
            test_metrics = {}
            if validation_set is not None:
                self.eval()
                for callback in self.callbacks:
                    callback.on_test_begin(logs={})
                test_metrics = self._feed_model(validation_set, validation=True)
                for callback in self.callbacks:
                    callback.on_test_end(logs=test_metrics)
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs={**train_metrics, **test_metrics})

            if self.stop_training:
                break

        for callback in self.callbacks:
            callback.on_train_end(logs={**train_metrics, **test_metrics})

    def _feed_model(self, generator, validation=False):
        prefix = ""
        if validation:
            prefix = "val_"
        metrics_vals = {prefix + "loss":0}
        for metric in self.metrics:
            metrics_vals[prefix + metric.__name__] = 0
        for batch, (dwi, next_dir, lengths) in enumerate(generator):
            metrics_vals_per_epoch = {}
            if not validation:
                self.optimizer.zero_grad()
                for callback in self.callbacks:
                    callback.on_train_batch_begin(batch, logs={"size":len(lengths), "batch":batch})
            else:
                for callback in self.callbacks:
                    callback.on_test_batch_begin(batch, logs={"size":len(lengths), "batch":batch})

            self.reset()
            pred_next_dir = self(dwi)

            mask = (torch.arange(dwi.shape[0])[None, :] < lengths[:, None]).transpose(0, 1).to(next_dir.device)

            pred_next_dir = pred_next_dir * mask[..., None]
            next_dir = next_dir * mask[..., None]
            loss = self.loss(pred_next_dir, next_dir)

            metrics_vals_per_epoch[prefix + "loss"] = loss
            for metric in self.metrics:
                metrics_vals_per_epoch[prefix + metric.__name__] = metric(pred_next_dir, next_dir)

            if not validation:
                loss.backward()
                self.optimizer.step()
                for callback in self.callbacks:
                    callback.on_train_batch_end(batch, logs=metrics_vals_per_epoch)
            else:
                for callback in self.callbacks:
                    callback.on_test_batch_begin(batch, logs=metrics_vals_per_epoch)

            for key in metrics_vals_per_epoch:
                metrics_vals[key] += len(lengths)*metrics_vals_per_epoch[key]

        for key in metrics_vals:
            metrics_vals[key] *= 1/len(generator.dataset)
        return metrics_vals

    def collate_fn(self, el):
        '''
        Padds batch of variable length
        '''
        device = next(self.parameters()).device
        inputs = [torch.flatten(t[0], start_dim=1) for t in el]
        outputs = [torch.flatten(t[1], start_dim=1) for t in el]

        lengths = torch.tensor([t[0].shape[0] for t in el])

        inputs = torch.nn.utils.rnn.pad_sequence(inputs).to(device)
        outputs = torch.nn.utils.rnn.pad_sequence(outputs).to(device)

        return inputs, outputs, lengths
