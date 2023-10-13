#!/usr/bin/env python
# coding: utf-8

# In[12]:


#!pip install streamlit


# In[9]:


import streamlit as st


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.metrics import roc_auc_score,roc_curve
import warnings
warnings.filterwarnings("ignore")


# In[13]:


st.title("Model Deployment: Diabetes Data")
st.sidebar.header("User Input Parameters")


# In[15]:


def user_parameters():
    Pregnancies=st.sidebar.number_input("Insert no. of Pregnancies")
    Glucose=st.sidebar.number_input("Insert Glucose level")
    BloodPressure=st.sidebar.number_input("Insert Blood Pressure")
    Skin_Thickness=st.sidebar.number_input("Insert skin thickness")
    Insulin=st.sidebar.number_input("Insert Insulin level")
    BMI=st.sidebar.number_input("Insert BMI")
    DiabetesPedigreeFunction=st.sidebar.number_input("insert Diabetes Pedigree Function")
    Age=st.sidebar.number_input("Insert Age")
    
    data={"Pregnancies":Pregnancies,
         "Glucose":Glucose,
         "BloodPressure":BloodPressure,
         "SkinThickness":Skin_Thickness,
         "Insulin":Insulin,
         "BMI":BMI,
         "DiabetesPedigreeFunction":DiabetesPedigreeFunction,
         "Age":Age}
    
    features=pd.DataFrame(data,index=[0])
    return features


# In[16]:


df=user_parameters()
st.subheader("Inputed Parameters are : ")
st.write(df)


# In[17]:


df=pd.read_csv("C:\\Users\\adite\\OneDrive\\Documents\\diabetes.csv")
X=df.drop(["Outcome"],axis=1)
y=df[["Outcome"]]


# In[18]:


smote=SMOTE()
balanced_X,balanced_y=smote.fit_resample(X,y)


# In[19]:


pca=PCA()
pca_df=pca.fit_transform(balanced_X)
final_pca_df=pd.DataFrame({"Pregnancies":pca_df[:,0],"Glucose":pca_df[:,1],"BloodPressure":pca_df[:,2],
                      "SkinThickness":pca_df[:,3],"Insulin":pca_df[:,4],"Outcome":balanced_y.Outcome})


# In[20]:


final_x=final_pca_df.drop({"Outcome"},axis=1)
final_y=final_pca_df[["Outcome"]]


# In[21]:


scaler=StandardScaler()
scaled_X=scaler.fit_transform(final_x)
scaled_y=scaler.fit_transform(final_y)


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(scaled_X,scaled_y,test_size=0.3,random_state=12)


# In[54]:


SVM_model=SVC(kernel="rbf",C=0.5)
SVM_model.fit(X_train,y_train)


# In[55]:


SVM_train_prediction=SVM_model.predict(X_train)
SVM_train_ac=accuracy_score(y_train,SVM_train_prediction)
SVM_train_ac=SVM_train_ac*100
SVM_train_ac


# In[56]:


SVM_train_prediction[0]


# In[57]:


st.subheader('Predicted Result')
st.write('Yes, He\She has a diabetes' if SVM_train_prediction[0]==1.0 else 'No, He\She does not has a diabetes')


# In[58]:


st.subheader('Prediction is')
st.write(SVM_train_prediction[0])


# In[ ]:




