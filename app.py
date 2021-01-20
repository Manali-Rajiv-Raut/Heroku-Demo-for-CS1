# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 01:11:01 2021

@author: Manali
"""

#from module_2_preprocessing import Data_Preprocessing
import joblib
import pickle
import numpy as np
#import pandas as pd
import nltk
import sklearn
#from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
clf = joblib.load("saved_model/RF_regressor_model_.pkl") 

with open('saved_model/vectorizer_sit', 'rb') as f:             
            vectorizer_sit = pickle.load(f)
        
with open('output_files/feature_dict', 'rb') as f:             
    feature_dict = pickle.load(f)
        
with open('output_files/Features', 'rb') as f:             
    new_features = pickle.load(f)
    

@app.route('/')
def home():
    return render_template('index.html')


def new_vec_with_support_values(vec,feature_dict , new_features):
    new_train_vec = np.zeros((vec.shape[0],vec.shape[1]))
    for row in range(len(vec)):
        for col in range(len(vec[row])):
            if vec[row][col] != 0.0:
                #print(X_test[row][col],"==>",col)
                val = feature_dict[new_features[col]]
                new_train_vec[row][col] = val 
    return new_train_vec

@app.route('/predict',methods=['POST'])
def predict():
    
    sentence = [x for x in request.form.values()]
    #dp = Data_Preprocessing()
    #sentence_preprocessed = dp.preprocess_text(sentence)
    vec  = vectorizer_sit.transform(sentence).toarray()
    #print("vec",vec.shape)
    new_vec = new_vec_with_support_values(vec,feature_dict , new_features)
    #print("new_vec" , new_vec.shape)
        
    y_pred = clf.predict(new_vec)
        
    if y_pred !=0.0 :
        #print("Predicted label : Trustworthy")
        return render_template('index.html', prediction_text='The person tweeted this \n {} is \n Trustworthy'.format(sentence))
    else:
        #print("Predicted label : Not Trustworthy") 
        return render_template('index.html', prediction_text='The person tweeted this \n {} is \n Not Trustworthy'.format(sentence))

if __name__ == "__main__":
    app.run(debug=True)




