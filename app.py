#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install joblib


# In[2]:

import gunicorn 
#! pip install gunicorn


# In[3]:


#!python -m pip install flask


# In[4]:


#import sklearn.ensemble.forest


# In[5]:


import sklearn
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


# In[7]:


import joblib


# In[8]:


app = Flask(__name__)

#eclf = joblib.load("saved_model/eclf_ML_Model_.pkl")
model =joblib.load("RF_regressor_model_.pkl")   
vectorizer_sit = joblib.load('vectorizer_sit.pkl')

with open('feature_dict', 'rb') as f:   
    feature_dict = pickle.load(f)

with open('features', 'rb') as f:      
    new_features = pickle.load(f)


# In[9]:


@app.route('/')
def home():
    return render_template("index.html")


# In[10]:


@app.route("/predict",methods = ["POST"])
def predict():
    # for rendering results on HTML GUI
    
    text = [x for x in request.form.values()]
    preprocessed_txt = vectorizer_sit.transform(text).toarray()
    new_vec = np.zeros((preprocessed_txt.shape[0],preprocessed_txt.shape[1]))

    for row in range(len(preprocessed_txt)):
        for col in range(len(preprocessed_txt[row])):
            if preprocessed_txt[row][col] != 0.0:
                #print(X_test[row][col],"==>",col)
                val = feature_dict[new_features[col]]
                new_vec[row][col] = val
    
    y_predicted = model.predict(new_vec)
    
    if y_predicted > 0.0:
        return render_template('index.html',prediction_text= " The person tweeted this {}  is Trustworthy".format(text))
    else:
         return render_template('index.html',prediction_text= " The person tweeted this {}  is Not Trustworthy".format(text))


# In[11]:


from app import app

if __name__ == "__main__":
    app.run(debug = True)


# In[ ]:




