#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:48:41 2020

Run this with Python 3 to change/downgrade pickle protocol, as protocol 4 and above are only compatible with Python 3.
@author: mubariz
"""
import pickle

with open('DIR/Somefile_Protocol3.pkl', 'rb') as handle:
    b = pickle.load(handle)
    
with open('DIR/Somefile_Protocol2.pkl', 'wb') as handle:
    pickle.dump(b, handle, protocol=2)
