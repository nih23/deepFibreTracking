'the training routine'
import logging
from datetime import datetime
import sys
import signal
import socket

import torch
from torch.utils import data as dataL
import torch.optim as optim
import torch.nn as nn
import argparse
#import horovod.torch as hvd

from src.dataset import Dataset
from src.model import get_random_model, ModelLSTM
from src.util import radians_loss, strfdelta
from src.logger import Logger


def getdatasets(path, training_part=0.9):
    """Retrieves a dataset from given path and splits them randomly in train and test data.

    Arguments:
    path: the dataset path
    training_part: float indicating the percentage of training data. default: 0.9 (90%)
    """
    sets = Dataset(path)
    train_len = int(training_part*len(sets))
    test_len = len(sets) - train_len
    (train_split, test_split) = torch.utils.data.random_split(sets, (train_len, test_len))
    return sets, train_split, test_split

def train_model(model, state, sets, logger):
    """Trains given model with given parameters, datasets and logs the progress.

    Arguments:
    model: the model to train
    state: the parameters for training
    sets: the training and test set as tuple (train, test)
    logger: the logger to track and save progress
    """
    (base_set, training_set, validation_set) = sets
    params = {'batch_size': state.batch_size, 'shuffle': True}
    training_generator = dataL.DataLoader(training_set, **params)
    validation_generator = dataL.DataLoader(validation_set, **params)
    optimizer = optim.Adam(model.parameters(), lr=state.learning_rate)
    no_epochs = 1000
    start_time = datetime.now()
    for epoch in range(1, no_epochs + 1):
        model.train()
        data = (base_set, training_set, training_generator)
        training_loss = feed_model(model, data, state, optimizer=optimizer)
        model.eval()
        data = (base_set, validation_set, validation_generator)
        with torch.no_grad():
            validation_loss = feed_model(model, data, state)
        (loss, rad_loss, p_stop_loss) = validation_loss
        logging.info("Epoch {:5d}/{:<5d} - loss: {:6.5f} rad_loss :\
{:6.5f} pStop_loss: {:6.5f} | elapsed time: {}"\
            .format(epoch, no_epochs, loss, rad_loss, p_stop_loss,\
            strfdelta((datetime.now() - start_time), '{H:02}:{M:02}:{S:02}')))
        logger.save_epoch(training_loss, validation_loss)
    logger.finish_param_file()

def feed_model(model, data, state, optimizer=None):
    """Passes data through given model and optimizes model according to optimizer if given.

    Arguments:
    model: the model
    data: a tuple containing (base_set, validation_set, validation_generator)
    state: the parameters for training
    optimizer: the optimizer to call. default: None
    """
    mse_loss = torch.nn.MSELoss()
    (base_set, current_set, generator) = data
    complete_p_stop_loss = 0
    complete_rad_loss = 0
    for batch_dwi, batch_next_direction, batch_continue_tracking, batch_padded_lengths in generator:
        if optimizer is not None:
            optimizer.zero_grad()
        if isinstance(model, ModelLSTM): # important -  reset is necessary
            model.reset()
        batch_len = len(batch_dwi)
        batch_dwi = batch_dwi.transpose(0, 1)
        batch_next_direction = batch_next_direction.transpose(0, 1)
        batch_continue_tracking = batch_continue_tracking.transpose(0, 1)
        batch_padded_lengths = batch_padded_lengths.cuda()
        if state.crop:
            batch_dwi = batch_dwi[:, :, 1300:1400]
        pred_next_direction, pred_continue_tracking = model(batch_dwi) # predict model
        mask = \
            (torch.arange(base_set.get_max_length())[None, :].cuda()\
             < batch_padded_lengths[:, None].long())\
            .transpose(0, 1).unsqueeze(2).float() # masking loss for padded values

        batch_next_direction = batch_next_direction*mask.expand(base_set.get_max_length(),\
            batch_len, 3)
        batch_continue_tracking = batch_continue_tracking*mask
        pred_next_direction = pred_next_direction*mask.expand(base_set.get_max_length(),\
            batch_len, 3)
        pred_continue_tracking = pred_continue_tracking*mask
        # calculate loss
        rad_loss = radians_loss(batch_next_direction, pred_next_direction, mask)
        p_stop_loss = mse_loss(batch_continue_tracking, pred_continue_tracking)
        loss = rad_loss + p_stop_loss
        factor = (batch_len/len(current_set))
        complete_rad_loss = complete_rad_loss + factor * rad_loss.item()
        complete_p_stop_loss = complete_p_stop_loss + factor * p_stop_loss.item()
        # optimize
        if model.training:
            loss.backward()
            optimizer.step()
    return (complete_p_stop_loss + complete_rad_loss, complete_rad_loss, complete_p_stop_loss)

def wrapper():
    """The basic training wrapper. Trains specific model in current implementation.
    """
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Specific Model Training')
    parser.add_argument('-networktype', dest='layer_size', default=None, help='network type: LSTM/MLP [default: random]')
    parser.add_argument('-datatype', dest='dataset_type', default=None, help='dataset type: 3x/1x [default: random]')
    parser.add_argument('-bs', dest='batch_size', default=None, type=int, help='batch size used for training [default: random]')
    parser.add_argument('-lr', dest='learning_rate', default=None, type=float, help='learning rate used for training [default:random]')
    parser.add_argument('-layersize', dest='layer_size', default=None, type=int, help='layer size of the model [default: random]')
    parser.add_argument('-depth', dest='depth', default=None, type=int, help='depth of the model [default: random]')
    parser.add_argument('-dropout', dest='dropout', default=None, type=float, help='Dropout used for training. Between 0 and 1 [default: random]')
    parser.add_argument('-activationfunction', dest='activation_function_string', default='', help="Activation function to use. Options: Tanh/ReLU/lReLU [default: random]")
    args = parser.parse_args()
    args.activation_function = None
    if args.activation_function_string.lower() == "tanh":
        args.activation_function = nn.Tanh()
    if args.activation_function_string.lower() == "relu":
        args.activation_function = nn.ReLU()
    if args.activation_function_string.lower() == "lrelu":
        args.activation_function = nn.LeakyReLU()
    logging.basicConfig(filename='logs/node-{}-{}.log'.format(socket.gethostname(), 4 #hvd.rank()
    ),\
        level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    torch.cuda.set_device(0) #hvd.local_rank())
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    logging.info("Loading dataset")
    sets = getdatasets("cache/data_ijk.pt")
    logging.info("Loaded dataset")
    def signal_handler(*_args):
        if logger is not None:
            logger.remove()
        sys.exit()
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        (model, state) = get_random_model(depth = args.depth, layer_size = args.layer_size, network_type=args.network_type, dataset_type=args.dataset_type, dropout=args.dropout, batch_size=args.batch_size, learning_rate=args.learning_rate, activation_function=args.activation_function)
        logging.info("Retrieved model")
        torch.manual_seed(2343)
        torch.cuda.manual_seed(2343)
        logger = Logger(state, model)
        train_model(model, state, sets, logger)
        logging.info("Trained model")
    except RuntimeError as err:
        if 'out of memory' in str(err) or 'Allocator' in str(err):
            logging.warning('ran out of memory, generate new model')
            if logger is not None:
                logger.remove()
        else:
            raise err    
if __name__ == "__main__":
    wrapper()
