"""Callback module
"""
import time

class Callback(object):
    """Abstract base class used to build new callbacks.
    Attributes:
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    """

    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        Arguments:
            epoch: integer, index of epoch.
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Has keys `batch` and `size` representing the current batch
                number and the size of the batch.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.
        Subclasses should override for any actions to run.
        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict. Metric results for this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.
        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.
        Arguments:
            logs: dict. Currently no data is passed to this argument for this method
                but that may change in the future.
        """

class SimpleLogger(Callback):
    "A simple Logger to use"
    def _timetostr(self, seconds):
        seconds = int(seconds)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
        elif hours > 0:
            return '%dh%dm%ds' % (hours, minutes, seconds)
        elif minutes > 0:
            return '%dm%ds' % (minutes, seconds)
        else:
            return '%ds' % (seconds,)
    def on_train_begin(self, logs=None):
        print("############# Starting training ####################")
        print("# Epochs : {}".format(self.params["epochs"]))
        print("# Validation : {}".format(self.params["do_validation"]))
        print("# -- Training --")
        print("# Batch size : {}".format(self.params["batch_size"]))
        print("# Steps : {}".format(self.params["steps"]))
        print("# Samples : {}".format(self.params["samples"]))
        print("# -- Validation --")
        print("# Batch size : {}".format(self.params["val_batch_size"]))
        print("# Steps : {}".format(self.params["val_steps"]))
        print("# Samples : {}".format(self.params["val_samples"]))
        print("####################################################")

    def on_epoch_begin(self, epoch, logs=None):
        print("Epoch {}/{} --".format(epoch, self.params["epochs"]), end="\r")
        self.epoch = epoch
        self.start_time = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {}/{} - {} ".format(epoch, self.params["epochs"],  self._timetostr(time.monotonic() - self.start_time)), end="")
        for key in logs:
            print("-- {}: {:.6f}  ".format(key, logs[key]), end="")
        print()

    def on_train_batch_end(self, batch, logs=None):
        print("Epoch {}/{} -- {} - Train: Batch {}/{} -- loss: {:.2f}   ".format(self.epoch, self.params["epochs"], self._timetostr(time.monotonic() - self.start_time), batch, self.params["steps"], logs["loss"]), end="\r")

    def on_test_batch_end(self, batch, logs=None):
        print("Epoch {}/{} -- {} - Test: Batch {}/{} -- loss: {:.2f}   ".format(self.epoch, self.params["epochs"], self._timetostr(time.monotonic() - self.start_time), batch, self.params["val_steps"], logs["val_loss"]), end="\r")
    def set_params(self, params):
        self.params = params