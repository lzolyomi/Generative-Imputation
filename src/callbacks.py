import os
import pandas as pd
import tensorflow as tf
from datetime import datetime

import random


class Callbacks:
    """Object to make callbacks easy in model 
    """

    def __init__(self, args: dict):
        ### logging
        self.logging = args["log"] if "log" in args.keys() else False
        self.logarrays = {}  # store arrays for logs

        ### early stopping
        self.early_stop = args["early_stop"] if "early_stop" in args.keys() else False
        self.stagnate = 0
        self.best_loss = 1e9
        if self.early_stop:
            print("$$$ EARLY STOP ENABLED @threshold {}".format(self.early_stop))

        ### Saving generator
        self.gen_save = args["gen_save"] if "gen_save" in args.keys() else False
        self.generator = None  # stores generator

        ### Saving discriminator
        self.discr_save = args["discr_save"] if "discr_save" in args.keys() else False
        self.discriminator = None  # stores discriminator

        ### Range of random batches
        self.random_batches = (
            args["random_batches"] if "random_batches" in args.keys() else 1
        )

    def add_log(self, name: str, value: float):
        """Adds value to log

        Args:
            name (str): name of value (column name)
            value (float): value to be logged
        """
        if self.logging:  # only engage if logging enabled
            # check if name already exist
            if name not in self.logarrays.keys():
                self.logarrays[name] = []
            # add value to dictionary
            self.logarrays[name].append(value)

    def create_log(self, path=False):
        """Creates logfile

        Args:
            path (str, optional): If given, logfile is saved on path. Defaults to cwd/logs/.
        """
        if self.logging:
            keys = list(self.logarrays.keys())
            # check if all values have same length
            assert len(self.logarrays[keys[0]]) == len(self.logarrays[keys[-1]])

            if not path:  # create path if none given
                path = os.getcwd() + "/logs/"
            df = pd.DataFrame(
                self.logarrays
            )  # create dataframe from dictionary with logs
            now = datetime.now()
            if not os.path.isdir(path):
                # create dir if doesn't exist (defaults to logs/ directory)
                os.mkdir(path)
            ### create path to dump log file
            logpath = path + "train-run-{}-{}.csv".format(now.hour, now.minute)
            df.to_csv(logpath)
            print(f">>> Log file saved in {logpath}")

    def early(self, metric: float):
        """Checks if model performance stagnates

        Args:
            metric (float): metric to be used for checking performance

        Returns:
            bool: True if stagnates, engage early stopping
        """
        if self.early_stop:  # engage if enabled
            if self.stagnate > 10:  # 10 is not fine tuned
                return True

            if self.best_loss > (metric + self.early_stop):
                # if new best score found restart stagnate, update best observed
                self.stagnate = 0
                self.best_loss = metric
            else:
                # if no new best observed update stagnate
                self.stagnate = self.stagnate + 1

    def save_model(self, iter, generator, discr):
        """saves models 

        Args:
            iter (int): current number of epoch
            generator (keras.models): Generator model
            discr (keras.models): Discriminator model
        """
        if self.gen_save:
            if iter % self.gen_save == 0:
                # save model at every given epoch
                # (eg. gen_save=10 saves every 10th epoch)
                self.generator = tf.keras.models.clone_model(generator)

        if self.discr_save:
            if iter % self.discr_save == 0:
                self.discriminator = tf.keras.models.clone_model(discr)

    def batch_proportion(self):
        """Returns float [0,1] indicating proportion of batches to be used in that epochs
        Returns:
            float: proportion of batches
        """
        # if percentage of batches specified
        if type(self.random_batches) in [int, float]:
            # sanity checks
            assert self.random_batches > 0, "Random batch proportion should be (0,1]"
            assert self.random_batches <= 1, "Random batch proportion should be (0,1]"
            return float(self.random_batches)
        # if range specified
        elif type(self.random_batches) in [tuple, list]:
            assert (
                len(self.random_batches) == 2
            ), "random batch range should be length 2"
            # return random float within specified range
            return random.uniform(self.random_batches[0], self.random_batches[1])

