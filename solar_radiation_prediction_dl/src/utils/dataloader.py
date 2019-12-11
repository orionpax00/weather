import tensorflow as tf
import pandas as pd
import numpy as np

TRAIN_SPLIT = 0.9
START_INDEX = 0

class getData(object):

    def __init__(self, mainfile:str, features_to_consider:list, \
                    target:str, history:int, future_target:int, steps:int):
        self.MAIN_FILE_PATH = mainfile
        self.multivariate = True
        self.normalize = True
        self.df = pd.read_csv(self.MAIN_FILE_PATH)
        self.split = int(TRAIN_SPLIT * len(self.df))
        self.features_to_consider = features_to_consider
        self.target = target
        self.history = history
        self.steps = steps
        self.target_size = future_target
        self.start_index = START_INDEX
        self.end_index = self.split

    def _multivariate(self,for_validation=False, single_step = False):
        
        features = self.df[self.features_to_consider]
        features.index = self.df["index"]

        target = self.df[self.target]

        dataset = features.values
        data_mean = dataset[:self.split].mean(axis = 0)
        data_std = dataset[:self.split].std(axis=0)

        dataset = (dataset - data_mean)/data_std

        data = []
        labels = []

        self.start_index = self.start_index + self.history

        if for_validation:
            self.start_index = self.split + self.history
            self.end_index = len(dataset) - self.target_size
        
        for i in range(self.start_index, self.end_index):
            indices = range(i - self.history, i , self.steps)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i+self.target_size])

            else:
                labels.append(target[i:i+self.target_size])

        return np.array(data), np.array(labels)

    def call(self, for_validation=False, single_step = False, multivariate=True):

        if multivariate:
            if for_validation:
                return self._multivariate(for_validation=True, single_step = single_step)
            else:
                return self._multivariate(single_step = single_step)


