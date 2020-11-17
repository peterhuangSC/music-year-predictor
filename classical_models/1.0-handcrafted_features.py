#!/usr/bin/env python
# coding: utf-8

# # Hand-crafted features for GTZAN
# 
# > The goal of this notebook is to create several audio features descriptors for the GTZAN dataset, as proposed for many year as input for machine learning algorithms. We are going to use timbral texture based features and tempo based features for this. The main goal is to produce this features, classify and then compare with our proposed deep learning approach, using CNNs on the raw audio.

# In[1]:


import os
import librosa
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew


# In[2]:


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

import lightgbm as lgbm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# In[3]:


# Set the seed
np.random.seed(42)


# In[4]:


gtzan_dir = '../../../files/genres/'


# In[5]:


# Parameters
genres = dict()
for i in range(0, 61):
    genres[str(i)] = i


# In[6]:


def get_features(y, sr, n_fft = 1024, hop_length = 512):
    # Features to concatenate in the final dictionary
    features = {'centroid': None, 'roloff': None, 'flux': None, 'rmse': None,
                'zcr': None, 'contrast': None, 'bandwidth': None, 'flatness': None}
    
    # Count silence
    if 0 < len(y):
        y_sound, _ = librosa.effects.trim(y, frame_length=n_fft, hop_length=hop_length)
    features['sample_silence'] = len(y) - len(y_sound)

    # Using librosa to calculate the features
    features['centroid'] = librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['roloff'] = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['rmse'] = librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    features['contrast'] = librosa.feature.spectral_contrast(y, sr=sr).ravel()
    features['bandwidth'] = librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['flatness'] = librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel()
    
    # MFCC treatment
    mfcc = librosa.feature.mfcc(y, n_fft = n_fft, hop_length = hop_length, n_mfcc=13)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()
        
    # Get statistics from the vectors
    def get_moments(descriptors):
        result = {}
        for k, v in descriptors.items():
            result['{}_max'.format(k)] = np.max(v)
            result['{}_min'.format(k)] = np.min(v)
            result['{}_mean'.format(k)] = np.mean(v)
            result['{}_std'.format(k)] = np.std(v)
            result['{}_kurtosis'.format(k)] = kurtosis(v)
            result['{}_skew'.format(k)] = skew(v)
        return result
    
    dict_agg_features = get_moments(features)
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
    
    return dict_agg_features


# In[7]:


def read_process_songs(src_dir, debug = True):    
    # Empty array of dicts with the processed features from all files
    arr_features = []

    # Read files from the folders
    for x,_ in genres.items():
        folder = src_dir + str(int(x) + 1960) 

        print("processing year: ", x)
        
        for root, subdirs, files in os.walk(folder):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                
                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))
                
                # Append the result to the data structure
                features = get_features(signal, sr)
                features['genre'] = genres[x]
                arr_features.append(features)
    return arr_features


# In[8]:


get_ipython().run_cell_magic(u'time', u'', u'\n# Get list of dicts with features and convert to dataframe\nfeatures = read_process_songs(gtzan_dir, debug=False)')


# In[9]:


df_features = pd.DataFrame(features)


# In[10]:


df_features.shape


# In[11]:


df_features.head()


# In[12]:


df_features.to_csv('../data/gtzan_features.csv', index=False)


# In[13]:


X = df_features.drop(['genre'], axis=1).values
y = df_features['genre'].values


# ## Visualization
# 
# > Linear (and nonlinear) dimensionality reduction of the GTZAN features for visualization purposes

# In[14]:


# Standartize the dataset
#scale = StandardScaler()
#x_scaled = scale.fit_transform(X)


# In[15]:


# Use PCA only for visualization
#pca = PCA(n_components=20, whiten=True)
#x_pca = pca.fit_transform(x_scaled)
#print("cumulative explained variance ratio = {:.4f}".format(np.sum(pca.explained_variance_ratio_)))


# In[16]:


# Use LDA only for visualization
#lda = LDA()
#x_lda = lda.fit_transform(x_scaled, y)


# In[17]:


# Using tsne
#tsne = TSNE(n_components=2, verbose=1, learning_rate=250)
#x_tsne = tsne.fit_transform(x_scaled)


# In[18]:


#plt.figure(figsize=(18, 4))
#plt.subplot(131)
#plt.scatter(x_pca[:,0], x_pca[:,1], c=y)
#plt.colorbar()
#plt.title("Embedded space with PCA")

#plt.subplot(132)
#plt.scatter(x_lda[:,0], x_lda[:,1], c=y)
#plt.colorbar()
#plt.title("Embedded space with LDA")

#plt.subplot(133)
#plt.scatter(x_tsne[:,0], x_tsne[:,1], c=y)
#plt.colorbar()
#plt.title("Embedded space with TSNE")
#plt.show()


# ## Classical Machine Learning

# In[19]:


# Helper to plot confusion matrix -- from Scikit-learn website
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.show()


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


# ### Linear Regression

# In[21]:


params = {
}

pipe_lr = Pipeline([
    ('scale', StandardScaler()),
    ('var_tresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('cls', LinearRegression())
])

grid_lr = GridSearchCV(LinearRegression(), params, scoring='neg_mean_absolute_error', n_jobs=6, cv=5)
grid_lr.fit(X_train, y_train)


# In[22]:


preds = grid_lr.predict(X_test)
print("linear regression best score on validation set (accuracy) = {:.4f}".format(grid_lr.best_score_))
print("linear regression  score on test set (accuracy) = {:.4f}".format(mean_absolute_error(y_test, preds)))


# ### Decision Tree

# In[25]:


params = {
    "cls__criterion": ["mse", "mae"],
    "cls__splitter": ["best", "random"],
}

pipe_cart = Pipeline([
    ('var_tresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('cls', DecisionTreeRegressor())
])

grid_cart = GridSearchCV(pipe_cart, params, scoring='neg_mean_absolute_error', n_jobs=6, cv=5)
grid_cart.fit(X_train, y_train)


# In[26]:


preds = grid_cart.predict(X_test)
print("decision tree best score on validation set (accuracy) = {:.4f}".format(grid_cart.best_score_))
print("decision tree best score on test set (accuracy) = {:.4f}".format(mean_absolute_error(y_test, preds)))


# ### Random Forest

# In[27]:


params = {
    "cls__n_estimators": [100, 250, 500, 1000],
    "cls__criterion": ["mse", "mae"],
    "cls__max_depth": [5, 7, None]
}

pipe_rf = Pipeline([
    ('var_tresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('cls', RandomForestRegressor())
])

grid_rf = GridSearchCV(pipe_rf, params, scoring='neg_mean_absolute_error', n_jobs=6, cv=5)
grid_rf.fit(X_train, y_train)


# In[28]:


preds = grid_rf.predict(X_test)
print("random forest best score on validation set (accuracy) = {:.4f}".format(grid_rf.best_score_))
print("random forest best score on test set (accuracy) = {:.4f}".format(mean_absolute_error(y_test, preds)))


# ### SVR

# In[29]:


params = {
    "cls__C": [0.5, 1, 2, 5],
    "cls__kernel": ['rbf', 'linear', 'sigmoid'],
}

pipe_svr = Pipeline([
    ('scale', StandardScaler()),
    ('var_tresh', VarianceThreshold(threshold=(.8 * (1 - .8)))),
    ('cls', SVR())
])

grid_svr = GridSearchCV(pipe_svr, params, scoring='neg_mean_absolute_error', n_jobs=6, cv=5)
grid_svr.fit(X_train, y_train)


# In[30]:


preds = grid_svr.predict(X_test)
print("svr best score on validation set (accuracy) = {:.4f}".format(grid_svr.best_score_))
print("svr best score on test set (accuracy) = {:.4f}".format(mean_absolute_error(y_test, preds)))


# ## Results and save the model

# In[31]:


#cm = confusion_matrix(y_test, preds)
#classes = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 'country', 'pop', 'blues', 'reggae', 'rock']


# In[32]:


#plt.figure(figsize=(10,10))
#plot_confusion_matrix(cm, classes, normalize=True)


# In[33]:


from sklearn.externals import joblib


# In[35]:


joblib.dump(grid_lr, "../models/pipe_lr.joblib")
joblib.dump(grid_cart, "../models/pipe_cart.joblib")
joblib.dump(grid_rf, "../models/pipe_rf.joblib")
joblib.dump(grid_svr, "../models/pipe_svr.joblib")

