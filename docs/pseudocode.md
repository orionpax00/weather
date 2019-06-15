<u>**Training Algorithm**</u><br>
1. **Input:** {[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]} eka **Dataset D1**
2. **Output:** Ridge Regressor Error
3. **Algorithm:**<br>
    Choose **Models** [xgb1, xgb2, ..., xgbk] as XGBoost Forest **&** a **DNN**<br>
    Train all models on dataset.<br>
    Create a new {[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [x(k+1)1, x(k+1)2, ..., x(k+1)n]} **Dataset D2**<br>
    **Where:**<br>
    ---> First Column = xgb1.predict(**D1**) and so on...<br>
    Split **D2** into ***D2_train*** and ***D2_test*** Dataset<br>
    Train **Ridge Regressor** on ***D2_train*** <br>
    validate on ***D2_test***
