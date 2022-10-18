#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import random
import sys


## Package
import glob 
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow
py.init_notebook_mode(connected=True)


## Keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical


## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm


# In[2]:


from keras import regularizers
import os


# In[3]:


import librosa
from librosa import display

data, sampling_rate = librosa.load('/Users/abhiram/Audio_Speech_Actors_01-24_Actor_01_03-01-01-01-01-01-01.wav')
print(data)
print(sampling_rate)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
from librosa import display
import os
import pandas as pd
import glob 
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate)


# 

# In[5]:



import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys


sr,x = scipy.io.wavfile.read('/Users/abhiram/Audio_Speech_Actors_01-24_Actor_01_03-01-01-01-01-01-01.wav')

## Parameters: 10ms step, 30ms window
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin

window = np.hamming(nwin)

## will take windows x[n1:n2].  generate
## and loop over n2 such that all frames
## fit within the waveform
nn = range(nwin, len(x), nstep)

X = np.zeros( (len(nn), nfft//2) )

for i,n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i,:] = np.log(np.abs(z[:nfft//2]))

plt.imshow(X.T, interpolation='nearest',
    origin='lower',
    aspect='auto')

plt.show()


# In[6]:


import time

path = '/Users/abhiram/Desktop/Audio_Speech_Actors_01-24/Actor_01'
lst = []

start_time = time.time()

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1 
        arr = mfccs, file
        lst.append(arr)
      # If the file is not valid, skip it
      except ValueError:
        continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))


# In[7]:


# Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
X, y = zip(*lst)


# In[8]:


import numpy as np
X = np.asarray(X)
y = np.asarray(y)


X.shape, y.shape


# In[9]:


import joblib

X_name = 'X.joblib'
y_name = 'y.joblib'
save_dir="/Users/abhiram/Desktop/Saved_Models"

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))


# In[10]:


X = joblib.load("/Users/abhiram/Desktop/Saved_Models/X.joblib")
y = joblib.load('/Users/abhiram/Desktop/Saved_Models/y.joblib')


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=11)


# In[12]:


from sklearn.tree import DecisionTreeClassifier


# In[13]:


dtree = DecisionTreeClassifier()


# In[14]:


dtree.fit(X_train, y_train)


# In[15]:


predictions = dtree.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[18]:


rforest = RandomForestClassifier(criterion="gini", max_depth=10, max_features="log2", 
                                 max_leaf_nodes = 100, min_samples_leaf = 3, min_samples_split = 20, 
                                 n_estimators= 22000, random_state= 5)


# In[19]:


rforest.fit(X_train, y_train)


# In[20]:


predictions = rforest.predict(X_test)


# In[21]:


print(classification_report(y_test,predictions))


# In[22]:


x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)


# In[23]:


x_traincnn.shape, x_testcnn.shape


# In[24]:


from tensorflow import keras
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)


# In[25]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[26]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=1000, validation_data=(x_testcnn, y_test))


# In[27]:


predictions =model.predict(x_testcnn) #x_testcnn


# In[ ]:





# In[28]:


new_Ytest = y_test.astype(int)


# In[29]:


new_Ytest


# In[30]:


# Loss 
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[31]:


from sklearn.metrics import confusion_matrix
import numpy as np
predictions=np.argmax(predictions, axis=1)
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)
# 0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised
l=0
m=0
n=0
o=0
p=0
q=0
r=0
s=0
a=[]
for i in matrix:
    for j in i:
        if(j==0):
            l=l+1
        elif(j==1):
            m=m+1
        elif(j==2):
            n=n+1
        elif(j==3):
            o=o+1
        elif(j==4):
            p=p+1
        elif(j==5):
            q=q+1
        elif(j==6):
            r=r+1
        elif(j==7):
            s=s+1
a.append(m)
a.append(n)
a.append(o)
a.append(p)
a.append(q)
a.append(r)
a.append(s)
t=max(a)
if(t==m):
    print("THE USER IS CALM")
elif(t==n):
    print("THE USER IS HAPPY")

elif(t==o):
    print("THE USER IS SAD")
elif(t==p):
    print("THE USER IS ANGRY")
elif(t==q):
    print("THE USER IS FEARFUL")
elif(t==r):
    print("THE USER IS DISGUSTED")
elif(t==s):
    print("THE USER IS SURPRISED")
            
            
        


# In[32]:


model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = '/Users/abhiram/Desktop/Saved_Models'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[33]:


from tensorflow import keras

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import CustomObjectScope

from tensorflow.keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

    loaded_model = load_model('/Users/abhiram/Desktop/Saved_Models/Emotion_Voice_Detection_Model.h5')
    loaded_model.summary()


# In[34]:


loss, acc = loaded_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




