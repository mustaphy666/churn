import streamlit as st
import pandas as pd
import pickle
import numpy as np
st.title('CHURN PREDICTION APP')
st.write('**This app predict whether a customer has churned or not by tuning the user inputs parameters**')
st.subheader('User inputs') 
def user():
    gender=st.selectbox('Gender',('Male','Female'))
    subscription_type=st.selectbox('Subscription Type',('Basic','Premium','Standard'))
    contract_length=st.selectbox('Contract Length',('Annual','Monthly','Quarterly'))
    age=st.slider('Age',18,65)
    tenure=st.slider('Tenure',1,60)
    usage_frequency=st.slider('Usage Frequency',0,30)
    support_calls=st.slider('Support Calls',0,10)
    payment_delay=st.slider('Payment Delay',0,30)    
    total_spent=st.slider('Total Spent',100,1000)
    last_interaction=st.slider('Last Interaction',0,30)
    data={'Gender':gender,'Subscription Type':subscription_type,
          'Contract Length':contract_length,'Age':age,'Tenure':tenure,'Usage Frequency':usage_frequency,
          'Support Calls':support_calls,'Payment Delay':payment_delay,
          'Total Spend':total_spent,'Last Interaction':last_interaction}
    features=pd.DataFrame(data,index=[0])
    return features
df=user()
testing_data=pd.read_csv('testing.csv')
testing_data=testing_data.drop(columns=['Churn','CustomerID'])
testing=pd.concat([df,testing_data],axis=0)
encode=['Gender','Subscription Type','Contract Length']
for c in encode:
    dummy=pd.get_dummies(testing[c],prefix=c)
    testing=pd.concat([testing,dummy],axis=1)
    del testing[c]
testing=testing[:1]
load=pickle.load(open('churn.pkl','rb'))
if st.button('Predict',type='primary'):
    pred=load.predict(testing)
    if pred.item() == 1:
        st.warning('The customer has churned.')
    else:
        st.success('The customer has not churned.')