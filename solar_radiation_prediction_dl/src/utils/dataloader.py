import tensorflow as tf
import pandas as pd
import numpy as np

TRAIN_SPLIT = 0.9
START_INDEX = 0

class getData(object):

    def __init__(self, mainfile:str, features_to_consider:list, \
                 history:int, future_target:int, steps:int):
        self.MAIN_FILE_PATH = mainfile
        self.multivariate = True
        self.normalize = True
        self.df = pd.read_csv(self.MAIN_FILE_PATH)
        self.split = int(TRAIN_SPLIT * len(self.df))
        self.features_to_consider = features_to_consider
        self.history = history
        self.steps = steps
        self.target_size = future_target
        self.start_index = START_INDEX
        self.end_index = self.split

    def _multivariate(self,for_validation=False, single_step = False):
        
        features = self.df[self.features_to_consider]
        # features.index = self.df["index"]

        dataset = features.values

        #normalize using text data
        data_mean = dataset[:self.split].mean(axis = 0)
        data_std = dataset[:self.split].std(axis=0)

        #normalize using the whole
        # data_mean = dataset.mean(axis = 0)
        # data_std = dataset.std(axis=0)

        dataset = (dataset - data_mean)/data_std

        target = dataset[:,-1]

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



    def testdata(self,single_step):

        features = self.df[self.features_to_consider]
        dataset = features.values

        #normalize using text data
        data_mean = dataset[:self.split].mean(axis = 0)
        data_std = dataset[:self.split].std(axis=0)

        #normalize using the whole
        # data_mean = dataset.mean(axis = 0)
        # data_std = dataset.std(axis=0)

        dataset = (dataset - data_mean)/data_std

        data = []
        labels = []

        target = dataset[:,-1]
        if single_step:
            start_index = len(dataset) - 20 * self.history - self.target_size
            end_index = len(dataset) - self.target_size
            
            for i in range(start_index, end_index):
                indices = range(i - self.history, i , self.steps)
                data.append(dataset[indices])
                # data.append(dataset[start_index:end_index])
                labels.append(target[i+self.target_size])
        
        else:
            start_index = len(dataset) - self.history - self.target_size
            end_index = len(dataset) - self.target_size
            data.append(dataset[start_index:end_index])
            labels.append(target[-self.target_size:])
        
        test_data = tf.data.Dataset.from_tensor_slices((data,labels))
        test_data = test_data.batch(1)

        return test_data, data_mean, data_std


