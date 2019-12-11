import tensorflow as tf
import pandas as pd

TRAIN_SPLIT = 0.8


class getData:

    def __init__(self,mainfile:str):
        self.MAIN_FILE_PATH = mainfile
        self.multivariate = True
        self.normalize = True
        self.df = pd.read_csv(self.MAIN_FILE_PATH)
        self.split = int(TRAIN_SPLIT * len(self.df))

    def _multivariate(self, features_to_consider:list):
        
        features = df[features_to_consider]
        features.index = self.df["index"]

        dataset = features.values
        data_mean = dataset[:self.split].mean(axis = 0)
        data_std = dataset[:self.split].std(axis=0)

        dataset = (dataset - data_mean)/data_std

        data = []
        label = []

        start_index = start_index + history_size

        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i - history_size, i , steps)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i+target_size])

            else:
                labels.append(target_size[i:i+target_size])

        return np.array(data), np.array(labels)

    def call(self,multivariate=True, start_index, target, start_index, end_index, history_size, target_size, steps, single_step = False):

        if multivariate:
            return self._multivariate(start_index, target, start_index, end_index, history_size, target_size, steps, single_step= False)


