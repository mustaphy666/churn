import pandas as pd 
import numpy as np
testing_data=pd.read_csv('.\\testing.csv')
encode=['Gender','Subscription Type','Contract Length']
for c in encode:
    dummy=pd.get_dummies(testing_data[c],prefix=c)
    testing_data=pd.concat([testing_data,dummy],axis=1)
    del testing_data[c]  
train=testing_data.drop(columns=['Churn','CustomerID'])
target=testing_data['Churn']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,target,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=10,n_estimators=40,n_jobs=-1,random_state=42)
model.fit(x_train,y_train)
import pickle
pickle.dump(model,open('churn.pkl','wb'))
