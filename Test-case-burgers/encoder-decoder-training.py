# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt 

from scipy.sparse import diags

from scipy.sparse.csgraph import reverse_cuthill_mckee

import scipy.sparse as sp

from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout
from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf
from tensorflow import keras
import pickle
import keras
from keras.layers import LeakyReLU
from matplotlib import cm

###########################################


x_train_small = np.load('Burgers/data/u_ens_32.npy').reshape((-1, 32, 32,1))-1.
x_train_large = np.load('Burgers/data/u_ens_128.npy').reshape((-1, 128, 128,1))-1.

x_test_small = np.load('Burgers/data/u_ens_32_bis2.npy').reshape((-1, 32, 32,1))-1.
x_test_large = np.load('Burgers/data/u_ens_128_bis2.npy').reshape((-1, 128, 128,1))-1.


##################################################

# define the large encoder 

n_large = 128

input_img_large = Input(shape=(n_large,n_large, 1))
 
x = Convolution2D(4, (4, 4), activation='relu', padding='same')(input_img_large)
x = MaxPooling2D((4, 4), padding='same')(x)
x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
flat = Flatten()(x)
flat = LeakyReLU(alpha=0.3)(flat)
encoded = Dense(15)(flat)

encoder_large = Model(input_img_large, encoded)

####################################################
# define the low-dimensional encoder

n_small = 32

input_img_small = Input(shape=(n_small,n_small, 1))
 
x = Convolution2D(4, (2, 2), activation='relu', padding='same')(input_img_small)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(4, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
flat = Flatten()(x)
flat = LeakyReLU(alpha=0.3)(flat)
encoded = Dense(15)(flat)

encoder_small = Model(input_img_small, encoded)

#####################################################

#define the decoder

decoder_input= Input(shape=(15,))

decoded = Dense(30,activation='relu')(decoder_input)
decoded = Dense(8*8*8)(decoded)
decoded = LeakyReLU(alpha=0.3)(decoded)
x = Reshape((8, 8, 8))(decoded)

x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(4, (4, 4), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
decoded = Convolution2D(1, (10, 10), activation='sigmoid', padding='same')(x)

decoder_large = Model(decoder_input, decoded)

#######################################################

#define the state-in-state-out autoencoder

auto_input_large = Input(batch_shape=(None,128,128, 1))
encoded_large = encoder_large(auto_input_large)
decoded_large = decoder_large(encoded_large)

autoencoder_large = Model(auto_input_large, decoded_large)
autoencoder_large.compile(optimizer='adam', loss='mse')

#define the observation-in-state-out autoencoder

auto_input_small = Input(batch_shape=(None,32,32, 1))

encoded_small = encoder_small(auto_input_small)
decoded_large = decoder_large(encoded_small)

autoencoder_small = Model(auto_input_small, decoded_large)
autoencoder_small.compile(optimizer='adam', loss='mse')


########################################################

#alternating training

l = range(0,1601,16)
large_loss = []
large_valloss = []
small_loss = []
small_valloss = []

for i in range(200):
  history = autoencoder_large.fit(x_train_large, x_train_large,validation_data=(x_test_large, x_test_large),epochs=1, batch_size=16,shuffle=True)
  history2 = autoencoder_small.fit(x_train_small, x_train_large[l,:,:,:],validation_data=(x_test_small, x_test_large[l,:,:,:]),epochs=1, batch_size=16,shuffle=True)

  large_loss.append(history.history['loss'])
  large_valloss.append(history.history['val_loss'])
  small_loss.append(history2.history['loss'])
  small_valloss.append(history2.history['val_loss'])
  
  
plt.plot(range(5,100),np.array(large_loss).ravel()[5:] ,label = 'high train')
plt.plot(range(5,100),np.array(large_valloss).ravel()[5:],'r',label = 'high val' )
plt.plot(range(5,100),np.array(small_loss).ravel()[5:], 'g' ,label = 'low train')
plt.plot(range(5,100),np.array(small_valloss).ravel()[5:], 'k',label = 'low val' )
plt.xlabel('Epochs',fontsize = 18)
plt.ylabel('MSE loss',fontsize = 18)
plt.legend()
#plt.savefig('drive/MyDrive/Burgers/figure/multiloss.eps', format='eps', bbox_inches='tight')


