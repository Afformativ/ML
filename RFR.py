    #import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from csv import writer
from sklearn.model_selection import train_test_split

# Import the DMS dataset
data = pd.read_csv('ML_Data_trim_MeOH_LogP.csv',encoding="latin1")
data.head()
#Un-normalized data
inputs = data.drop(labels=['Compound', 'SMILES', 'BW_CCS', 'LogP_pred', 'LogP_predRange'], axis = 'columns')
target = data['LogP_pred']

ts = 0.20
t = 1

with open('MeOH_WS_output_ts80_rs64_NoCCS.csv', 'a', newline='') as f_object:

    #create error curve
    for t in range(t):
        
        #split the dataset into training and test datasets
        x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size = ts, random_state = 64)
        
        #create a Random Forest regressor object from Random Forest Regressor class
        RFReg = RandomForestRegressor(n_estimators = 500, random_state = 64) #n-estimators is the number of trees.
        
        #fit the random forest regressor with training data represented by X_train and y_train
        RFReg.fit(x_train, y_train)
        
        #predicted height from test dataset wrt Random Forest Regression
        y_predict_rfr = RFReg.predict((x_test))
        y_train_pred = RFReg.predict((x_train))
              
        #Model evaluation using R-square from Random Forest Regression
        r_square = metrics.r2_score(y_test, y_predict_rfr)
        #print('R-square error associated with Random Forest Regression is:', r_square)
        
        MAE = metrics.mean_absolute_error(y_test, y_predict_rfr)
        #print('MAE is:', MAE)
        
        params = [t, ts, r_square, MAE]
        
        writer_object = writer(f_object) 
        writer_object.writerow(params)
        
        #comment out when generating correlation plot data.
        #ts = ts - 0.01
        
        #uncomment below and set t=1 to generate correlation plot data.
        a = np.array(y_train)
        b = np.array(y_train_pred)
        c = np.array(y_test)
        d = np.array(y_predict_rfr)
        
        df = pd.DataFrame({"y_train" : a, "y_train_pred" : b})
        df.to_csv("WS_MeOH_train_predictions_rs64_NoCCS.csv", mode='a', index=False)
        
        df = pd.DataFrame({"y_test" : c, "y_predict_rfr" : d})
        df.to_csv("WS_MeOH_test_predictions_rs64_NoCCS.csv", mode='a', index=False)

f_object.close()