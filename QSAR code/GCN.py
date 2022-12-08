# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:24:36 2022

@author: TYJ
"""

#Data Processing
import tensorflow as tf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import random
import copy
import scipy.sparse as sp
from collections import defaultdict


data = pd.read_csv('mPGES dataset.csv')
data_copy = copy.deepcopy(data)
smi_total= data_copy['mol'] = data_copy['SMILES'].apply(Chem.MolFromSmiles)
label_total= data_copy['label'].values.reshape(-1,1)


max_atom_n = 0
atom_species = defaultdict(int)
atom_degree = defaultdict(int)
atom_H_n = defaultdict(int)
atom_valence = defaultdict(int)

for smi in smi_total:
    mol = Chem.MolFromSmiles(smi.strip())
    
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    max_atom_n = max(adj.shape[0], max_atom_n)
    
    for atom in mol.GetAtoms():
        atom_species[atom.GetSymbol()] += 1
        atom_degree[atom.GetDegree()] += 1
        atom_H_n[atom.GetTotalNumHs()] += 1
        atom_valence[atom.GetImplicitValence()] += 1 

    
for inf in (max_atom_n, atom_degree, atom_H_n, atom_valence, atom_species):
    print(inf)
    

def atom_feature(atom):
    features =  list(map(lambda s: int(atom.GetDegree() == s), [1, 2, 3, 4])) + \
                list(map(lambda s: int(atom.GetTotalNumHs() == s), [0, 1, 2, 3])) + \
                list(map(lambda s: int(atom.GetImplicitValence() == s), [0, 1, 2, 3,])) + \
                list(map(lambda s: int(atom.GetSymbol() == s), ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br','I',])) + \
                [atom.GetIsAromatic()]

    return np.array(features)


num_features = len(atom_feature(atom))
print(num_features)

def smi_to_feat_adj(smiles_list, num_features):
    all_adj = []
    all_features = []
    all_index = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi.strip())
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        adj = sp.csr_matrix(np.array(adj, dtype='int64')).tocsr()

        feature = []
        for atom in mol.GetAtoms():
            feature.append( atom_feature(atom) ) 

        all_features.append(np.array(feature))
        all_adj.append(adj)
        all_index.append(len(feature))
    
    all_index = np.repeat(np.arange(len(all_index)), all_index)

    #to tensor
    all_adj = sp.block_diag(all_adj)
    all_adj = tf.SparseTensor(
        indices=np.array([all_adj.row, all_adj.col]).T,
        values=all_adj.data,
        dense_shape=all_adj.shape    
    )
    
    all_features = np.vstack(all_features)
    all_features = tf.convert_to_tensor(all_features)
    
    all_index  = tf.convert_to_tensor(all_index)
    return all_features, all_adj, all_index

num_train = 441
num_validation = 110
num_test = 184

smi_train = smi_total[0:num_train]
label_train = label_total[0:num_train]
smi_validation = smi_total[num_train:(num_train+num_validation)]
label_validation = label_total[num_train:(num_train+num_validation)]
smi_test = smi_total[(num_train+num_validation):]
label_test = label_total[(num_train+num_validation):]

def data_batchs(smi_list, label, batch_size):
    label = tf.convert_to_tensor(label)
    i = 0
    X, A, I, y = [], [], [], []
    
    while i < len(smi_list):

        batch_features, batch_adj, batch_index = smi_to_feat_adj(smi_list[i:i+batch_size], num_features)
        X.append(batch_features)
        A.append(batch_adj)
        I.append(batch_index)
        y.append(label[i:i+batch_size])

        i += batch_size
    
    return X, A, I, y

X_train, A_train, I_train, label_train = data_batchs(smi_train, label_train, 25)
X_validation, A_validation, I_validation, label_validation = data_batchs(smi_validation, label_validation, 500)
X_test, A_test, I_test, label_test = data_batchs(smi_test,label_test, 500)

#develop model
from tensorflow.keras.layers import Input, Dense, Layer, Dropout
from tensorflow.keras import activations, initializers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

tf.__version__

class Readout(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.dense = Dense(units=self.units, 
                             activation=None, 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros',)    

    def call(self, input_X, input_I):
        output_Z = self.dense(input_X)
        output_Z = tf.math.segment_sum(output_Z, input_I)
        output_Z = tf.nn.sigmoid(output_Z)
        return output_Z    


class GatedGCN(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.inp_dim = int(input_shape[-1])
        if(self.units != self.inp_dim):
            self.dense_i = Dense(
                units=self.units, 
                activation=None, 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                bias_initializer='zeros',)

        self.dense_n = Dense(units=self.units, 
                             activation=None, 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros',)    
        
        self.dense_gate_n = Dense(units=self.units, 
                             activation=None, 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros',)
        
        self.dense_gate_i = Dense(units=self.units, 
                             activation=None, 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             bias_initializer='zeros',)

    def call(self, input_X, input_A):
        new_X = self.dense_n(input_X)
        new_X = tf.sparse.sparse_dense_matmul(input_A, new_X)
        
        X1 = self.dense_gate_i(input_X)
        X2 = self.dense_gate_n(new_X)
        gate_coefficient = tf.nn.sigmoid(X1 + X2)

        if(self.units != self.inp_dim):
            input_X = self.dense_i(input_X)
            
        output_X = tf.multiply(new_X, gate_coefficient) + tf.multiply(input_X, 1.0-gate_coefficient)        
        
        return output_X
    
num_features=21

num_layer = 4
hidden_dim1 = 64
hidden_dim2 = 256
init_lr = 0.0001

X = Input(shape=(num_features,))
A = Input(shape=(None,), sparse=True)
I = Input(shape=(), dtype=tf.int64)

h = X

for i in range(num_layer):
    h = GatedGCN(units=hidden_dim1)(h, A)

h = Readout(units=hidden_dim2)(h, I) 

h = Dense(units=hidden_dim2, use_bias=True, activation='relu')(h)
h = Dense(units=hidden_dim2, use_bias=True, activation='tanh')(h)
h = Dropout(0.5)(h)
Y_pred = Dense(units=1, use_bias=True)(h)

model = Model(inputs=[X, A, I], outputs=Y_pred)
optimizer = Adam(lr=init_lr)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

#train and test
@tf.function(experimental_relax_shapes=True)
def train_step(x, a, i, y):
    with tf.GradientTape() as tape:
        predictions = model([x, a, i], training=True)
        loss = MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

@tf.function(experimental_relax_shapes=True)
def test_step(x, a, i, y):
    predictions = model([x, a, i], training=False)
    loss = MSE(y, predictions)
    return tf.reduce_mean(loss)

@tf.function(experimental_relax_shapes=True)
def train_step(x, a, i, y):
    with tf.GradientTape() as tape:
        predictions = model([x, a, i], training=True)
        loss = MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)

@tf.function(experimental_relax_shapes=True)
def test_step(x, a, i, y):
    predictions = model([x, a, i], training=False)
    loss = MSE(y, predictions)
    return tf.reduce_mean(loss)

epochs = 50

try:
    for epoch_n in range(epochs):
        epoch_loss = 0
        batch_n = 0
        for x, a, i, y in zip(X_train, A_train, I_train, label_train):
            loss = train_step(x, a, i, y)
            epoch_loss += loss
            batch_n += 1

            if batch_n == len(X_train):

                val_loss = 0
                val_n = 0
                for x, a, i, y in zip(X_validation, A_validation, I_validation, label_validation):
                    val_loss += test_step(x, a, i, y)
                    val_n += 1
                print(f'Epoch:{epoch_n}  MSE Loss:{epoch_loss/len(X_train):.3f}   val_Loss:{val_loss/val_n:.3f}')
                epoch_loss = 0
                batch_n = 0

                # shuffle
                c = list(zip(X_train, A_train, I_train, label_train))
                random.shuffle(c)
                X_train, A_train, I_train, label_train = zip(*c)

except:
    print('Early stoping')
    
test_n, test_loss = 0, 0
for x, a, i, y in zip(X_test, A_test, I_test, label_test):
    test_loss += test_step(x, a, i, y)
    test_n += 1
print(f'test_Loss:{test_loss/test_n:.3f}')

def predict_real_comparison(X, A, I,label):
    reals = np.array([])
    predictions = np.array([])

    for x, a, i, y in zip(X, A, I, label):
        pred = model([x, a, i], training=False)
        reals = np.concatenate((reals, y.numpy().reshape(-1)),axis=0)
        predictions = np.concatenate((predictions, pred.numpy().reshape(-1)),axis=0)
        
    return reals, predictions

train_real, train_prediction = predict_real_comparison(X_train, A_train, I_train, label_train)
test_real, test_prediction = predict_real_comparison(X_test, A_test, I_test, label_test)
validation_real, validation_prediction = predict_real_comparison(X_validation, A_validation, I_validation, label_validation)
