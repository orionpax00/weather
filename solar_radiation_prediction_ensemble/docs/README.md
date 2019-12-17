# Stacking Classifier For GHI Prediction

This repository contains python code for building a staking classifier of FDN and Xgboost for GHI Prediction.

## Stacking
Stacking is an ensemble learning technique to combine multiple classification models via a meta-classifier. The individual classification models are trained based on the complete training set; then, the meta-classifier is fitted based on the outputs -- meta-features -- of the individual classification models in the ensemble. The meta-classifier can either be trained on the predicted class labels or probabilities from the ensemble.

![img](stacking.png)

## Results
Using the one FDN with 12 Xgboosts the RSME was tremendiously decresed from **106.46** to **10.11** and keeps on increase one incresing the number of classifier. [here](https://github.com/orionpax00/weather/blob/cnnlstm/solar_radiation_prediction_ensemble/src/models/Stacked_classifier_dnn_xgboost.ipynb)

![img](results.png)


## Model Architecture and Parameters
### Xgboost
Parameter | Value
--- | ---
learning_rate|0.4
base_score|0.5 
max_depth|8
n_estimators|10
booster|gbtree 
objective|reg:linear
colsample_bylevel|1
colsample_bytree|1 
gamma|0 
importance_type|gain
max_delta_step|0
min_child_weight|1
missing|None
n_jobs|1
nthread|None
random_state|0
reg_alpha|0
reg_lambda|1
scale_pos_weight|1
seed|None
silent|True
subsample|1

### Deep Neural Network
![img](deeppara.png)

## Training
![img](training.png)