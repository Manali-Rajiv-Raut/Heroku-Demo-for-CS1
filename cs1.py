#!/usr/bin/env python
# coding: utf-8

# In[11]:


from module_2_preprocessing import Data_Preprocessing


# In[12]:


import joblib
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
#import pandas as pd


# In[13]:


class Find_Trustworthiness():
    
    def loading_files_models(self):
        rf_reg = joblib.load("saved_model/RF_regressor_model_.pkl") 
        vectorizer_sit = joblib.load('saved_model/vectorizer_sit.pkl')
        
        with open('output_files/feature_dict', 'rb') as f:             
            feature_dict = pickle.load(f)
        
        with open('output_files/Features', 'rb') as f:             
            new_features = pickle.load(f)
            
        return rf_reg , vectorizer_sit , feature_dict , new_features
    
    def new_vec_with_support_values(self,vec,feature_dict , new_features):
        new_train_vec = np.zeros((vec.shape[0],vec.shape[1]))
        for row in range(len(vec)):
            for col in range(len(vec[row])):
                if vec[row][col] != 0.0:
                    #print(X_test[row][col],"==>",col)
                    val = feature_dict[new_features[col]]
                    new_train_vec[row][col] = val 
        return new_train_vec
        
    
    def predict(self,sentence):
        rf_reg , vectorizer_sit , feature_dict , new_features = self.loading_files_models()
        
        dp = Data_Preprocessing()
        sentence_preprocessed = dp.preprocess_text(sentence)
        vec  = vectorizer_sit.transform(sentence_preprocessed).toarray()
        #print("vec",vec.shape)
        new_vec = self.new_vec_with_support_values(vec,feature_dict , new_features)
        #print("new_vec" , new_vec.shape)
        
        y_pred = rf_reg.predict(new_vec)
        
        if y_pred !=0.0 :
            print("Predicted label : Trustworthy")
        else:
            print("Predicted label : Not Trustworthy")              


# In[14]:


ft = Find_Trustworthiness()
#ft.predict("Run a marathon in under two hours. Impossible? Not for Nike (@Nike). Last May, the company brought three of the best runners on the planet together in Italy to set a new record in a closed-door marathon that was broadcast live on Twitter.")


# In[15]:


sentence = 'For #NationalVegetarianWeek, the English food supplier Tesco (@Tesco) shared a range of delicious vegetarian recipes in customized Moments'


# In[16]:


ft.predict(sentence)


# In[ ]:




