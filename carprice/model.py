import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('car data.csv')

outlier_cols = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven']

def remove_outliers_iqr(data, column):
  q1,q2,q3 = np.percentile(data[column],[25,50,75])
  print("q1,q2,q3 is :",q1,q2,q3)
  IQR = q3-q1
  print("IQR is :" ,IQR)
  lower_limit = q1-(1.5*IQR)
  upper_limit = q3+(1.5*IQR)
  data[column]=np.where(data[column]>upper_limit,upper_limit,data[column]) # Capping the upper limit
  data[column]=np.where(data[column]<lower_limit,lower_limit,data[column]) # Flooring the lower limit
  
for column in outlier_cols:
  remove_outliers_iqr(df,column)
df.drop(columns=['Car_Name'], inplace=True)
current_year = datetime.now().year

le= LabelEncoder()
# le_trans = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])  # Label Encoding for Seller_Type
df['Transmission'] = le.fit_transform(df['Transmission'])  # Label Encoding for Transmission
df = pd.get_dummies(df, columns=['Seller_Type'], drop_first=True) 

with open('Fuel_Type.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('Transmission.pkl', 'wb') as f:
    pickle.dump(le, f)

x=df.drop('Selling_Price',axis=1)
y=df['Selling_Price']

standadisation=StandardScaler()
x_scale=standadisation.fit_transform(x)
# Coverting to Dataframe
x=pd.DataFrame(x_scale)

with open('scaling.pkl', 'wb') as f:
    pickle.dump(standadisation, f)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =42,test_size=0.33)

model = RandomForestRegressor()
model.fit(x_train, y_train)

pickle.dump(model,open('model.pkl','wb'))