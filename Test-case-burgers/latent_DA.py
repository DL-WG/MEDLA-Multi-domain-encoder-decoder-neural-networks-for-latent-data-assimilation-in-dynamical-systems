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
# linear DA solution function

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

################################################
#load encoder and decoder models
encoder_small =  keras.models.load_model('Burgers/encoder_small_32.h5')
encoder_large = keras.models.load_model('Burgers/encoder_large_128.h5')
decoder_large = keras.models.load_model('Burgers/decoder_large_128.h5')

#################################################

x_train_small = np.load('drive/MyDrive/Burgers/data/u_ens_32.npy').reshape((-1, 32, 32,1))-1.
x_train_large = np.load('drive/MyDrive/Burgers/data/u_ens_128.npy').reshape((-1, 128, 128,1))-1.

x_test_small = np.load('drive/MyDrive/Burgers/data/u_ens_32_bis2.npy').reshape((-1, 32, 32,1))-1.
x_test_large = np.load('drive/MyDrive/Burgers/data/u_ens_128_bis2.npy').reshape((-1, 128, 128,1))-1.

####################################################
# define the error covariance matrix

import math
def get_index_2d (dim,n): #get caratesian coordinate
    j=n % dim
    j=j/1. #float( i)
    i=(n-j)/dim
    return (i,j)# pourquoi float?

def Balgovind(dim,L):

    L = L*1.
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r = math.sqrt((a1-a2)**2+(b1-b2)**2)*1.

            sub_B[i,j]=(1.+r/L)*(math.exp(-r/L))
            
##################################################

n_large = 128
n_small = 32

sigma = 0.2

COV = np.dot(np.dot(np.diag(np.sqrt(np.abs(x_test_small[50,:,:,0].ravel()))*sigma),Balgovind(32,5)),np.diag(np.sqrt(np.abs(x_test_small[50,:,:,0].ravel()))*sigma))

u_decoded = x_test_small[50,:,:,0] #+  np.random.multivariate_normal(np.zeros(n_small*n_small), COV).reshape(n_small,n_small)

#u_decoded = x_test_small[50,:,:,0]+  np.random.multivariate_normal(np.zeros(n_small*n_small), np.diag(x_test_small[50,:,:,0].ravel()*sigma**2)).reshape(n_small,n_small)

x = np.linspace(0,2,n_small)     #Coordinate Along X direction
y = np.linspace(0,2,n_small)     #Coordinate Along Y direction

#2d temporaray array where we copy our velocity field
fig = plt.figure(figsize=(11,7),dpi=100)          #Initializing the figure
ax  = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)


ax.plot_surface(X,Y,u_decoded,cmap=cm.viridis,rstride=1,cstride=1)
#ax.plot_surface(X,Y,v[:],cmap=cm.viridis,rstride=1,cstride=1)

#ax.set_title('Velocity Field')
#ax.set_xlabel('X',fontsize = 16)
#ax.set_ylabel('Y',fontsize = 16)
ax.set_zlabel('Velocity',fontsize = 16)


####################################################################


import cv2
import numpy as np

#img = x_test_small[50,:,:,0]
img = u_decoded
full_cubic = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
full_linear = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

plt.imshow(full_linear)

#######################################################################


encode_cubic = encoder_large.predict(full_cubic.reshape(1,128,128,1))
encode_linear = encoder_large.predict(full_linear.reshape(1,128,128,1))
#encode_MEDLA = encoder_small.predict(x_test_small[50,:,:,0].reshape(1,32,32,1))
encode_MEDLA = encoder_small.predict(u_decoded.reshape(1,32,32,1))
xb = encoder_large.predict(x_test_large[50*12,:,:,0].reshape(1,128,128,1))

xa_cubic,_ = VAR_3D(xb,encode_cubic,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
xa_linear,_ = VAR_3D(xb,encode_linear,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
xa_MEDLA,_ = VAR_3D(xb,encode_MEDLA,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))

#u_g = ground_truth[-1,:].reshape(50,50)

ax.plot_surface(X,Y,u_decoded,cmap=cm.viridis,rstride=1,cstride=1)

##################################################################

u_decoded = xa_MEDLA

x = np.linspace(0,2,n_large)     #Coordinate Along X direction
y = np.linspace(0,2,n_large)     #Coordinate Along Y direction

#2d temporaray array where we copy our velocity field
fig = plt.figure(figsize=(11,7),dpi=100)          #Initializing the figure
ax  = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)


ax.plot_surface(X,Y,u_decoded,cmap=cm.viridis,rstride=1,cstride=1)
#ax.plot_surface(X,Y,v[:],cmap=cm.viridis,rstride=1,cstride=1)

#ax.set_title('Velocity Field')
#ax.set_xlabel('X Spacing')
#ax.set_ylabel('Y Spacing')
#ax.set_zlabel('Velocity')

# 270+u Points

#u_g = ground_truth[-1,:].reshape(50,50)

ax.plot_surface(X,Y,u_decoded,cmap=cm.viridis,rstride=1,cstride=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

##################################################################################

# Test robustness with error

##################################################################################


import cv2

x_true = x_test_large[50*16,:,:,0]

sigma = 0.01
obs = x_test_small[50,:,:,0].ravel() +  np.random.multivariate_normal(np.zeros(n_small*n_small), np.diag(x_test_small[50,:,:,0].ravel()*sigma**2))
obs = obs.reshape(n_small,n_small)

full_cubic = cv2.resize(obs, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
full_linear = cv2.resize(obs, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

encode_cubic = encoder_large.predict(full_cubic.reshape(1,128,128,1))
encode_linear = encoder_large.predict(full_linear.reshape(1,128,128,1))
encode_MEDLA = encoder_small.predict(x_test_small[50,:,:,0].reshape(1,32,32,1))
xb = encoder_large.predict(x_test_large[50*12,:,:,0].reshape(1,128,128,1))

xa_cubic,_ = VAR_3D(xb,encode_cubic,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
xa_linear,_ = VAR_3D(xb,encode_linear,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
xa_MEDLA,_ = VAR_3D(xb,encode_MEDLA,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))

Hxa_cubic = decoder_large.predict(xa_cubic.reshape(1,15))
Hxa_linear = decoder_large.predict(xa_linear.reshape(1,15))
Hxa_MEDLA = decoder_large.predict(xa_MEDLA.reshape(1,15))

print(np.linalg.norm(Hxa_cubic.ravel()-x_true.ravel())/np.linalg.norm(x_true))
print(np.linalg.norm(Hxa_linear.ravel()-x_true.ravel())/np.linalg.norm(x_true))
print(np.linalg.norm(Hxa_MEDLA.ravel()-x_true.ravel())/np.linalg.norm(x_true))


####################################################################################


import cv2

x_true = x_test_large[50*16,:,:,0]

cubic_list= []
MEDLA_list= []
linear_list= []


cubic_std= []
MEDLA_std= []
linear_std= []

#sigma = 0.01

for sigma in np.arange(0.0,0.5,0.05):
  print(sigma)
  cubic_current=[]
  MEDLA_current=[]
  linear_current=[]
  for j in range(50):
    #obs = x_test_small[50,:,:,0].ravel() +  np.random.multivariate_normal(np.zeros(n_small*n_small), np.diag(x_test_small[50,:,:,0].ravel()*sigma**2))

    COV = np.dot(np.dot(np.diag(np.sqrt(np.abs(x_test_small[50,:,:,0].ravel()))*sigma),Balgovind(32,2)),np.diag(np.sqrt(np.abs(x_test_small[50,:,:,0].ravel()))*sigma))

    obs = x_test_small[50,:,:,0]+  np.random.multivariate_normal(np.zeros(n_small*n_small), COV).reshape(n_small,n_small)
    obs = obs.reshape(n_small,n_small)

    full_cubic = cv2.resize(obs, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    full_linear = cv2.resize(obs, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)

    encode_cubic = encoder_large.predict(full_cubic.reshape(1,128,128,1))
    encode_linear = encoder_large.predict(full_linear.reshape(1,128,128,1))
    encode_MEDLA = encoder_small.predict(obs.reshape(1,32,32,1))
    xb = encoder_large.predict(x_test_large[50*20,:,:,0].reshape(1,128,128,1))

    xa_cubic,_ = VAR_3D(xb,encode_cubic,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
    xa_linear,_ = VAR_3D(xb,encode_linear,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))
    xa_MEDLA,_ = VAR_3D(xb,encode_MEDLA,np.eye(15),0.2**2*np.eye(15)*xb,0.01**2*np.eye(15))

    Hxa_cubic = decoder_large.predict(xa_cubic.reshape(1,15))
    Hxa_linear = decoder_large.predict(xa_linear.reshape(1,15))
    Hxa_MEDLA = decoder_large.predict(xa_MEDLA.reshape(1,15))

    MEDLA_current.append(np.linalg.norm(Hxa_MEDLA.ravel()-x_true.ravel())/np.linalg.norm(x_true))
    cubic_current.append(np.linalg.norm(Hxa_cubic.ravel()-x_true.ravel())/np.linalg.norm(x_true))
    linear_current.append(np.linalg.norm(Hxa_linear.ravel()-x_true.ravel())/np.linalg.norm(x_true))


  print(MEDLA_current)
  MEDLA_list.append(np.mean(MEDLA_current))
  cubic_list.append(np.mean(cubic_current))
  linear_list.append(np.mean(linear_current))

  MEDLA_std.append(np.std(MEDLA_current))
  cubic_std.append(np.std(cubic_current))
  linear_std.append(np.std(linear_current))