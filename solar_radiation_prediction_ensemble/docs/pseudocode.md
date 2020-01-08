<hr>

**Algorithm** (Stacked Regressor XGBoost Forest and FDN)<br>

<hr>

**Input: D1** =  {[x<sub>11</sub>, x<sub>12</sub>, ..., x<sub>1n</sub>], [x<sub>21</sub>, x<sub>22</sub>, ..., x<sub>23</sub>], ..., [x<sub>m1</sub>, x<sub>m2</sub>, ..., x<sub>mn</sub>], [y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>n</sub>]} :  x<sub>ij</sub> <belongs to\> R<sup>p</sup>, the Dataset<br>
--->x<sub>k</sub> = {[x<sub>11</sub>, x<sub>12</sub>, ..., x<sub>1n</sub>], [x<sub>21</sub>, x<sub>22</sub>, ..., x<sub>23</sub>], ..., [x<sub>m1</sub>, x<sub>m2</sub>, ..., x<sub>mn</sub>]}:feature vectors<br>
--->y<sub>k</sub> = [y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>n</sub>] : GHI<br>
**Output:** Ridge Regressor Error
#### **Steps:**
1. Split dataset **D1** after shuffling into **T** (training dataset) and **F** (Forcast Dataset),**x**<sub>kt</sub>U **y**<sub>kt</sub> = **T**, **x**<sub>kf</sub>U **y**<sub>kf</sub> = **F**, **TUA**=**D1**<br>
  Giving<br>
  **x**<sub>kt</sub>: Training feature vectors <br>
  **x**<sub>kf</sub>: Forcast feature vectors <br>
  **y**<sub>kt</sub>: Training GHI vectors <br>
  **y**<sub>kf</sub>: Forcast GHI vectors <br>

2. Initiate a XGBoost model and a FDN, find the best models as **xgb1** and **fdn1** using Grid Search Cross Validation respectively with dataset **T**.
3. Initiate **k** XGBoost models taking parameters around **xgb1** parameters.
4. Create a new **Dataset D2** {[x<sub>11</sub>, x<sub>12</sub>, ..., x<sub>1n</sub>], [x<sub>21</sub>, x<sub>22</sub>, ..., x<sub>23</sub>], ..., [x<sub>(k+1)1</sub>, x<sub>(k+1)2</sub>, ..., x<sub>(k+1)n</sub>]} <br>
    **Where:**<br>
    ---> [x<sub>i1</sub>, x<sub>i2</sub>,..., x<sub>in</sub>] = xgbi.predict(**T**), i <belongs to\> [1,k]<br>
    ---> [x<sub>(k+1)1</sub>, x<sub>(k+1)2</sub>,..., x<sub>(k+1)n</sub>] = fdn1.predict(**T**)<br>
5. Initiate a Ridge Regressor **R1**
6. Train **R1** on **D2** to predict **y**<sub>kt</sub>
7. validate or every model on Dataset **F**
