# -*- coding: utf-8 -*-
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
#from sympy import Symbol, nsolve, solve
#from sympy.solvers import solve
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl
import time

from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout,LayerNormalization
from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf

import tensorflow as tf

from tensorflow.python.keras.layers import LSTM,LeakyReLU,RepeatVector,TimeDistributed
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow import keras
import pickle
import keras
from keras.layers import LeakyReLU
from keras import backend as K
from keras import optimizers


################################################################
# load SVD compressed data


test_index = list(range(1,1600,2))
train_index = list(set(range(1,1600)) - set(range(1,1600,2)))

x_train_inv = np.load('PREMIERE/data/compressed_obs_inv.npy').T[train_index,:]
x_train_square = np.load('PREMIERE/data/compressed_obs_square.npy').T[train_index,:]
x_train_state = np.load('PREMIERE/data/compressed_train_all.npy').T[train_index,:]

x_test_state = np.load('PREMIERE/data/compressed_train_all.npy').T[test_index,:]
x_test_square = np.load('PREMIERE/data/compressed_obs_square.npy').T[test_index,:]
x_test_inv = np.load('PREMIERE/data/compressed_obs_inv.npy').T[test_index,:]

max_inv = np.max(np.abs(x_train_inv))
max_square = np.max(np.abs(x_train_square))

##################################################################
# normalisation

x_train_inv = x_train_inv *np.max(np.abs(x_train_state))/max_inv
x_train_square = x_train_square *np.max(np.abs(x_train_state))/max_square
x_test_inv = x_test_inv *np.max(np.abs(x_train_state))/max_inv
x_test_square = x_test_square *np.max(np.abs(x_train_state))/max_square

print(np.max(np.abs(x_train_state)))
print(np.max(np.abs(x_train_square)))
print(np.max(np.abs(x_train_inv)))

print(np.max(np.abs(x_test_state)))
print(np.max(np.abs(x_test_square)))
print(np.max(np.abs(x_test_inv)))

###############################################################
# multi-domain encoder decoder

latent_dim = 30

input_state = keras.Input(shape=(x_train_state.shape[1],))


encoded = Dense(128)(input_state)
#encoded = LayerNormalization()(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
#encoded = Dropout(0.3)(encoded)
encoded = Dense(latent_dim)(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
encoder_state = keras.Model(input_state, encoded)

input_inv = keras.Input(shape=(x_train_inv.shape[1],))


encoded = Dense(128)(input_inv)
#encoded = LayerNormalization()(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
encoded = Dropout(0.3)(encoded)
encoded = Dense(latent_dim)(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
encoder_inv = keras.Model(input_inv, encoded)

input_square = keras.Input(shape=(x_train_square.shape[1],))


encoded = Dense(128)(input_square)
#encoded = LayerNormalization()(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
#encoded = Dropout(0.3)(encoded)
encoded = Dense(latent_dim)(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
encoder_square = keras.Model(input_square, encoded)

decoder_input= Input(shape=(latent_dim,))
decoded = Dense(128)(decoder_input)
decoded  = LeakyReLU(alpha=0.3)(decoded )
#decoded = Dropout(0.2)(decoded)
#decoded = Flatten()(decoded)
decoded  = Dense(x_train_state.shape[1])(decoded )
decoded  = LeakyReLU(alpha=0.3)(decoded )
decoder = keras.Model(decoder_input, decoded)

##############################################################
# train the multi-domain encoder

state_input = keras.Input(shape=(x_train_state.shape[1],))
encoded = encoder_state(state_input)
decoded = decoder(encoded)

EC_state = keras.Model(state_input, decoded)
EC_state.compile(optimizer='adam', loss='mse')


inv_input = keras.Input(shape=(x_train_inv.shape[1],))
encoded = encoder_inv(inv_input)
decoded = decoder(encoded)

EC_inv = keras.Model(inv_input, decoded)
EC_inv.compile(optimizer='adam', loss='mse')


square_input = keras.Input(shape=(x_train_square.shape[1],))
encoded = encoder_square(square_input)
decoded = decoder(encoded)

EC_square = keras.Model(square_input, decoded)
EC_square.compile(optimizer='adam', loss='mse')


x_train_square = x_train_square.reshape(x_train_square.shape[0],x_train_square.shape[1],1)
x_train_inv = x_train_inv.reshape(x_train_inv.shape[0],x_train_inv.shape[1],1)
x_train_state = x_train_state.reshape(x_train_state.shape[0],x_train_state.shape[1],1)

lr_state = [0.001]*100+[0.0005]*100+[0.0001]*100+[0.000001]*10
lr_inv = [0.001]*100+[0.0005]*100+[0.0001]*100+[0.000001]*10
lr_square = [0.001]*100+[0.0005]*100+[0.0001]*100+[0.000001]*10

state_loss = []
state_valloss = []
inv_loss = []
inv_valloss = []
square_loss = []
square_valloss = []

for i in range(1010):
  K.set_value(EC_state.optimizer.learning_rate, lr_state[i])
  history = EC_state.fit(x_train_state, x_train_state,epochs=1,validation_split=0.1, batch_size=64,shuffle=True)
  K.set_value(EC_inv.optimizer.learning_rate, lr_inv[i])
  history2 = EC_inv.fit(x_train_inv, x_train_state,epochs=1,validation_split=0.1, batch_size=64,shuffle=True)
  K.set_value(EC_square.optimizer.learning_rate, lr_square[i])
  history3 = EC_square.fit(x_train_square, x_train_state,epochs=1,validation_split=0.1, batch_size=64,shuffle=True)

  state_loss.append(history.history['loss'])
  state_valloss.append(history.history['val_loss'])
  inv_loss.append(history2.history['loss'])
  inv_valloss.append(history2.history['val_loss'])
  square_loss.append(history3.history['loss'])
  square_valloss.append(history3.history['val_loss'])

######################################################################

#evaluate reconstruction error

######################################################################



x_test_state = np.load('drive/MyDrive/PREMIERE/data/compressed_train_all.npy').T[test_index,:]

#x_train_inv = x_train_inv *np.max(np.abs(x_train_state))/max_inv
#x_train_square = x_train_square *np.max(np.abs(x_train_state))/max_square
#x_test_inv = x_test_inv *np.max(np.abs(x_train_state))/max_inv
#x_test_square = x_test_square *np.max(np.abs(x_train_state))/max_square

encoded_train = encoder_state.predict(x_train_state.reshape(x_train_state.shape[0],1000,1))
encoded_test = encoder_state.predict(x_test_state.reshape(x_test_state.shape[0],1000,1))
#np.linalg.norm( encoded_test - encoder_square.predict(x_test_square.reshape(800,800,1)))/np.linalg.norm(encoded_test)#

acc_train = np.linalg.norm( encoded_train - encoder_state.predict(x_train_state.reshape(x_train_state.shape[0],1000,1)))/np.linalg.norm(encoded_train)#
acc_test = np.linalg.norm( encoded_test - encoder_state.predict(x_test_state.reshape(x_test_state.shape[0],1000,1)))/np.linalg.norm(encoded_test)#

square_train = np.linalg.norm( encoded_train - encoder_square.predict(x_train_square.reshape(x_train_square.shape[0],800,1)))/np.linalg.norm(encoded_train)#
square_test = np.linalg.norm( encoded_test - encoder_square.predict(x_test_square.reshape(x_test_square.shape[0],800,1)))/np.linalg.norm(encoded_test)#

inv_train = np.linalg.norm( encoded_train - encoder_inv.predict(x_train_inv.reshape(x_train_inv.shape[0],800,1)))/np.linalg.norm(encoded_train)#
inv_test = np.linalg.norm( encoded_test - encoder_inv.predict(x_test_inv.reshape(x_test_inv.shape[0],800,1)))/np.linalg.norm(encoded_test)#


print(acc_train)
print(acc_test)

print(square_train)
print(square_test)

print(inv_train)
print(inv_test)

###############################################################################
u_pod_1000 = np.load('drive/MyDrive/PREMIERE/data/u_pod_all.npy')[:,:1000]

state_recon = decoder.predict(encoder_state.predict(x_train_state.reshape(x_train_state.shape[0],1000,1)))
inv_recon = decoder.predict(encoder_inv.predict(x_train_inv.reshape(x_train_inv.shape[0],800,1)))
square_recon = decoder.predict(encoder_square.predict(x_train_square.reshape(x_train_square.shape[0],800,1)))

acc_state_full = np.linalg.norm(alpha_ens_train.T - np.dot(u_pod_1000,state_recon.T))/np.linalg.norm(alpha_ens_train)
acc_inv_full = np.linalg.norm(alpha_ens_train.T - np.dot(u_pod_1000,inv_recon.T))/np.linalg.norm(alpha_ens_train)
acc_square_full = np.linalg.norm(alpha_ens_train.T - np.dot(u_pod_1000,square_recon.T))/np.linalg.norm(alpha_ens_train)


print(round(latent_state_full,5))
print(round(latent_inv_full,5))
print(round(latent_square_full,5))

print(round(acc_state_full,5))
print(round(acc_inv_full,5))
print(round(acc_square_full,5))

###############################################################################

fine tuning to align the two encoders

###############################################################################

encoded_train = encoder_state.predict(x_train_state.reshape(x_train_state.shape[0],1000,1))
encoded_test = encoder_state.predict(x_test_state.reshape(x_test_state.shape[0],1000,1))
encoder_square.compile(optimizer='adam', loss='mse')
#history_square = encoder_square.fit(x_train_square, encoded_train,epochs=1000,validation_split=0.05, batch_size=64,shuffle=True)

history_square = encoder_square.fit(x_train_square, encoded_train,epochs=100,validation_data=(x_test_square, encoded_test), batch_size=64,shuffle=True)

encoder_inv.compile(optimizer='adam', loss='mse')
history_inv = encoder_inv.fit(x_train_inv, encoded_train,epochs=50,validation_split=0.05, batch_size=64,shuffle=True)

encoded_test = encoder_state.predict(x_test_state.reshape(800,1000,1))
#np.linalg.norm( encoded_test - encoder_square.predict(x_test_square.reshape(800,800,1)))/np.linalg.norm(encoded_test)#
np.linalg.norm( encoded_train - encoder_square.predict(x_train_square.reshape(1600,800,1)))/np.linalg.norm(encoded_train)#

encoder_state.save('drive/MyDrive/PREMIERE/model/encoder_state_SVDAE_lr.h5')
encoder_inv.save('drive/MyDrive/PREMIERE/model/encoder_inv_SVDAE_lr.h5')
encoder_square.save('drive/MyDrive/PREMIERE/model/encoder_square_SVDAE_lr.h5')
decoder.save('drive/MyDrive/PREMIERE/model/decoder_state_SVDAE_lr.h5')
