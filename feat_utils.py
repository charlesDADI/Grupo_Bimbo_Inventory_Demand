# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:03:28 2016

@author: charles-abner
"""
import pandas as pd
def one_hot_encoding(df,cols,dummy_na=True):
    res = pd.get_dummies(df,dummy_na=True,columns=cols)
    return res