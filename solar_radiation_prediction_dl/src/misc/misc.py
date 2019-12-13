import os
import csv
import json

from tensorflow.keras.utils import plot_model

BASE_RESULT_DIR = "D:\\projects\\Solar_radiation_prediction\\solar_radiation_prediction_dl\\src\\results"



def log_prediction(model,data,name,date_time, data_mean, data_std):
    RESULT_FILE = os.path.join(BASE_RESULT_DIR , "report" , name+ "_" + date_time + ".csv")
    with open(RESULT_FILE,mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for x, y in data:


            X_ = list(x[0][:,-1].numpy())
            X = [i * data_std + data_mean for i in X_]
            X.append(y.numpy()[0] * data_std + data_mean)
            X.append(model.predict(x)[0][0] * data_std + data_mean )
            csv_writer.writerow(X)

def configLogger(data:dict, name, date_time):
    CONFIG_FILE = os.path.join(BASE_RESULT_DIR , "configs" , name+ "_" + date_time + ".json")   
    with open(CONFIG_FILE, 'w') as fp:
        json.dump(data, fp)
