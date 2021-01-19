# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 02:48:41 2021

@author: Manali
"""
import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={'Post Your Tweet':'Everyhing is possible'})

print(r.json())