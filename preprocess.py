# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:57:02 2016

@author: charles-abner
"""

"""
__file__
    preprocess.py
__description__
    This file preprocesses data.
__author__
    Charles-Abner DADI
    
"""

import sys
import cPickle
import numpy as np
import pandas as pd
sys.path.append("../")
sys.path.append("/home/charles-abner/works/")
from param_config import config
import feat_utils
from sklearn.preprocessing import LabelBinarizer
###############
## Load Data ##
###############
print("Load data...")
nsample=10000
dfTrain = pd.read_csv(config.original_train_data_path,nrows=nsample).fillna("")
dfTest = pd.read_csv(config.original_test_data_path,nrows=nsample).fillna("")
dfOriginal_cliente_tabla_data_path = pd.read_csv(config.original_cliente_tabla_data_path,nrows=nsample ).fillna("")
dfOriginal_producto_tabla_path = pd.read_csv(config.original_producto_tabla_path,nrows=nsample).fillna("")
dfOriginal_sample_submission_data_path = pd.read_csv(config.original_sample_submission_data_path,nrows=nsample).fillna("")
dfOriginal_town_state_data_path = pd.read_csv(config.original_town_state_data_path,nrows=nsample).fillna("")

# number of train/test samples
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

############################
## Descriptive Statistics ##
############################
print("Exploratory datasets...")

dfTrain.apply(lambda x: len(x.unique()))

######################
## Pre-process Data ##
######################
print("Pre-process data...")

## insert sample index
dfTrain["index"] = np.arange(num_train)
dfTest["index"] = np.arange(num_test)

## join train/test with town_state on Agencia_ID
dfTrain = pd.merge(dfTrain, dfOriginal_town_state_data_path , on='Agencia_ID')
dfTest = pd.merge(dfTest, dfOriginal_town_state_data_path , on='Agencia_ID')

## join train/test with producto_tabla on Producto_ID
dfTrain = pd.merge(dfTrain, dfOriginal_producto_tabla_path , on='Producto_ID')
dfTest = pd.merge(dfTest, dfOriginal_producto_tabla_path , on='Producto_ID')

## join train/test with producto_tabla on Producto_ID
dfTrain = pd.merge(dfTrain, dfOriginal_cliente_tabla_data_path , on='Cliente_ID')
dfTest = pd.merge(dfTest, dfOriginal_cliente_tabla_data_path , on='Cliente_ID')

def one_encoding(df,col,ids):
    for i in range(len(ids)):
        df[''.join((col, "_%d" % (ids[i])))] = 0
        df[''.join((col, "_%d" % (ids[i])))][dfTrain[col]==(ids[i])] = 1
    return df

## one-hot encode the Agencia_ID
unique_agencias_id = dfOriginal_town_state_data_path['Agencia_ID'].unique()
dfTrain = one_encoding(dfTrain,'Agencia_ID',unique_agencias_id)
dfTest = one_encoding(dfTest,'Agencia_ID',unique_agencias_id)

## one-hot encode the Producto_ID
unique_productos_id = dfOriginal_producto_tabla_path['Producto_ID'].unique()
dfTrain = one_encoding(dfTrain,'Producto_ID',unique_agencias_id)
dfTest = one_encoding(dfTest,'Producto_ID',unique_agencias_id)


colsNominalCat = ["Agencia_ID","Canal_ID","Ruta_SAK","Producto_ID"]
dfTrain = feat_utils.one_hot_encoding(dfTrain,colsNominalCat,True)        
dfTest = feat_utils.one_hot_encoding(dfTest,colsNominalCat,True)        


print("Done.")