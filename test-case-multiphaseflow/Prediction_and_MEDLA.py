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

true_field = np.load('PREMIERE/data/alpha_ens_um104fi15_dense.npy')[350,:].ravel()

#################################################################################

# load models 


#model = keras.models.load_model('drive/MyDrive/PREMIERE/model/LSTM_SVDAE_10_lr_bis2_every10_100epo.h5')

model = keras.models.load_model('drive/MyDrive/PREMIERE/model/LSTM_SVDAE_10_lr_bis2_every10.h5')

#x_test_inv = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_inv.npy').T[1600:2400,:]
#x_test_square = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_square.npy').T[1600:2400,:]
#x_test_state = np.load('drive/MyDrive/PREMIERE/data/compressed_train_all.npy').T[1600:2400,:]


x_test_inv = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_inv.npy').T[800:1600,:]
x_test_square = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_square.npy').T[800:1600,:]
x_test_state = np.load('drive/MyDrive/PREMIERE/data/compressed_train_all.npy').T[800:1600,:]

#x_test_inv = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_inv.npy').T[1600:2400,:]
#x_test_square = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_square.npy').T[1600:2400,:]
#x_test_state = np.load('drive/MyDrive/PREMIERE/data/compressed_train_all.npy').T[1600:2400,:]

x_test_inv = x_test_inv *np.max(np.abs(x_test_state))/max_inv
x_test_square = x_test_square *np.max(np.abs(x_test_state))/max_square

#x_test_state = np.load('drive/MyDrive/PREMIERE/data/compressed_train_all.npy').T[800:1600,:]

encoder_state = keras.models.load_model('drive/MyDrive/PREMIERE/model/encoder_state_SVDAE_lr.h5')
encoder_inv = keras.models.load_model('drive/MyDrive/PREMIERE/model/encoder_inv_SVDAE_lr.h5')
encoder_square = keras.models.load_model('drive/MyDrive/PREMIERE/model/encoder_square_SVDAE_lr.h5')
decoder = keras.models.load_model('drive/MyDrive/PREMIERE/model/decoder_state_SVDAE_lr.h5')

encoded_test = encoder_state.predict(x_test_state.reshape(800,1000,1))

initial_latent = encoded_test[100:110,:]
current_latent = np.copy(initial_latent)

latent_error = []

encoded_predict = np.copy(encoded_test)



###########################################################################

# online prediction of LSTM without data assimilation

for index in range(110,800,10):
  current_latent = model.predict(current_latent.reshape(1,10,30))
  encoded_predict[index:index+10,:] = current_latent.reshape(10,30)
  #print('xa_seq',np.linalg.norm(current_latent.ravel()-encoded_test[index:index+10,:].ravel()))
  
  latent_error.append(np.linalg.norm(current_latent.ravel()-encoded_test[index:index+10,:].ravel()))
  if index == 350:
    print(current_latent.shape)
    np.save('drive/MyDrive/PREMIERE/data/model_high_350.npy',np.dot(u_pod,decoder.predict(current_latent[:,0,:]).reshape(-1,1)).ravel())
    model_field = np.dot(u_pod,decoder.predict(current_latent[:,0,:]).reshape(-1,1)).ravel()
  #print(current_latent.reshape(10,30)[:,5])
  #current_latent = encoded_test[index:index+10,:]
  
  
plt.plot(latent_error)
plt.close()

#plot prediction vs truth in the latent space


plt.plot(encoded_predict[:800,19],label='prediction')
plt.plot(encoded_test[:800,19],label='latent truth')
plt.xlabel('time steps',fontsize=15)
plt.legend(fontsize=12)


#######################################################################
# prediction with the correction of MEDLA

# define the linear resolution function

def VAR_3D(xb,Y,H,B,R): #booleen=0 garde la trace
    # xb: priori, Y: observation, H: obs. matrix, B: priori estimate uncertainty
    # R: Measurement uncertainty
    dim_x = xb.size
    #dim_y = Y.size
    Y.shape = (Y.size,1)
    xb1=np.copy(xb)
    xb1.shape=(xb1.size,1)
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R))) #matrice de gain
    
    A=np.dot(np.dot((np.eye(dim_x)-np.dot(K,H)),B),np.transpose((np.eye(dim_x)-np.dot(K,H))))+np.dot(np.dot(K,R),np.transpose(K))
    vect=np.dot(H,xb1)
    xa=np.copy(xb1+np.dot(K,(Y-vect)))
    return xa.ravel(),A


obs_square = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_square.npy').T[800:1600,:]
obs_inv = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_inv.npy').T[800:1600,:]

#obs_square = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_square.npy').T[1600:2400,:]
#obs_inv = np.load('drive/MyDrive/PREMIERE/data/compressed_obs_inv.npy').T[1600:2400,:]

obs_inv = obs_inv *np.max(np.abs(x_test_state))/max_inv
obs_square = obs_square *np.max(np.abs(x_test_state))/max_square

obs_latent_inv = encoder_inv.predict(obs_inv.reshape(800,800,1))

obs_latent_square = encoder_square.predict(obs_square.reshape(800,800,1))

initial_latent = encoded_test[100:110,:]
current_latent = np.copy(initial_latent)
latent_error = []
#DA_index = list(range(200,210)) + list(range(300,310))

DA_index = [] 

for i in range(150, 800,100):
  DA_index += [i]
  DA_index += [i+10]
  #DA_index += [i+20]
encoded_predict = np.copy(encoded_test)
for index in range(110,800,10):
  current_latent = model.predict(current_latent.reshape(1,10,30))

  if index in DA_index:
    #obs = obs_latent_square[index:index+10,:]
    obs = obs_latent_inv[index:index+10,:]


    xa_seq,_ = VAR_3D(current_latent.ravel(),obs.ravel(),np.eye(300),0.2*np.eye(300),0.01*np.eye(300))
    print('obs',np.linalg.norm(obs.ravel()-encoded_test[index:index+10,:].ravel()))
    current_latent = xa_seq
    #debug
    #current_latent = encoded_test[index:index+10,:],
  print('xa_seq',np.linalg.norm(current_latent.ravel()-encoded_test[index:index+10,:].ravel()))
  latent_error.append(np.linalg.norm(current_latent.ravel()-encoded_test[index:index+10,:].ravel()))
  encoded_predict[index:index+10,:] = current_latent.reshape(10,30)

  if index == 350:
    current_latent = current_latent.reshape(1,10,30)
    np.save('drive/MyDrive/PREMIERE/data/inv_low_350.npy',np.dot(u_pod,decoder.predict(current_latent[:,0,:]).reshape(-1,1)).ravel())
    DA_field = np.dot(u_pod,decoder.predict(current_latent[:,0,:]).reshape(-1,1)).ravel()
  #xa_MEDLA,_ = VAR_3D(xb,encode_MEDLA,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))

plt.plot(encoded_predict[:800,16],label='prediction')
plt.plot(encoded_test[:800,16],label='latent truth')
plt.xlabel('time steps',fontsize=15)
plt.legend(fontsize=12)

plt.plot(encoded_predict[:800,19],label='prediction')
plt.plot(encoded_test[:800,19],label='latent truth')
plt.xlabel('time steps',fontsize=15)
plt.legend(fontsize=12)