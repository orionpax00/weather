import scipy
import pandas as pd
import numpy as np
import matplotlib as plt 
import seaborn as sns



class csvReader(object):
    """
    A :class: `csvReader will handle the loading of csv file 

    Extended Summary
    ----------------
    It will be the easy way to load data rather then the pd.read_csv() but in backend it used the same function

    Notes
    ----------------
    Nothing yet

    Parameters
    ----------------
    location: string() 
        location of file.

    Examples
    ----------------
    >>>reader = csvReader("./data/base/base.csv")
    >>>reader.filter()
    """
    
    def __init__(self, file_location:str, drop_rows = True):

        self.location = file_location
        self.drop_rows = drop_rows

    def _readfile(self):
        df = pd.read_csv(self.location)

        return df
    
    def filterdata(self):

        data = self._readfile()

        if self.drop_rows:
            data.






    
        