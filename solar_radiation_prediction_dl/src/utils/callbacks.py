import os

from tensorflow.keras import callbacks

BASE_RESULT_DIR = "D:\\projects\\Solar_radiation_prediction\\solar_radiation_prediction_dl\\src\\results"


def logger(model_name, date_time, csvlogger = True, tensorboard=True):


    callbacks_list = []

    if csvlogger:
        CSV_PATH = os.path.join(BASE_RESULT_DIR , "csvlogs" , model_name+ "_" + date_time + ".csv")
        csv_logger = callbacks.CSVLogger(CSV_PATH)
        callbacks_list.append(csv_logger)
    if tensorboard:
        TENSORBOARD_DIR = os.path.join(BASE_RESULT_DIR , "tensorboard_logs" , model_name+ "_" + date_time )
        tensorboard = callbacks.TensorBoard(log_dir=TENSORBOARD_DIR)
        callbacks_list.append(tensorboard)



    return callbacks_list