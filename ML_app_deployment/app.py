#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


# In[2]:


import sklearn.externals.joblib as extjoblib
import joblib


# In[5]:


app = Flask(__name__)

eclf = joblib.load("saved_model/eclf_ML_Model_.pkl")
model =joblib.load("saved_model/RF_regressor_model_.pkl")   
vectorizer_sit = joblib.load('saved_model/vectorizer_sit.pkl')

with open('output_files/feature_dict', 'rb') as f:   
    feature_dict = pickle.load(f)

with open('output_files/features', 'rb') as f:      
    new_features = pickle.load(f)


# In[6]:


@app.route('/')
def home():
    return render_template("index.html")


# In[7]:


@app.route("/predict",methods = ["POST"])
def predict():
    # for rendering results on HTML GUI
    
    text = [x for x in request.form.values()]
    preprocessed_txt = vectorizer_sit.transform(text).toarray()
    pred_label = eclf.predict(preprocessed_txt)
    
    new_vec = np.zeros((preprocessed_txt.shape[0],preprocessed_txt.shape[1]))

    for row in range(len(preprocessed_txt)):
        for col in range(len(preprocessed_txt[row])):
            if preprocessed_txt[row][col] != 0.0:
                #print(X_test[row][col],"==>",col)
                val = feature_dict[new_features[col]]
                new_vec[row][col] = val
    
    y_predicted = model.predict(new_vec)
    
    if y_predicted != 0.0:
        return render_template('index.html',prediction_text= " The Predicted Emotion is {} . \n The person is Trustworthy".format(pred_label))
    else:
        return render_template('index.html',prediction_text= " The Predicted Emotion is {} . \n The person is Not Trustworthy".format(pred_label))
    


# In[8]:


if __name__ == "__main__":
    app.run(debug = True)


# In[ ]:




