# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:03:39 2017

@author: super
"""

import os
import pickle
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

from feature import FeatureExtractor


class VehicleFeatureClassifier(object):
    
    def __init__(self, clf=None):
        if clf is None:
            self.clf = LinearSVC()
        else: self.clf = clf
        self.fe = FeatureExtractor()
        self.scaler = None
        self.valid_acc = 0
    
    def load_features_and_labels(self, car_path, noncar_path):
        '''Load images and extract features, then make training dataset.'''
        car_feat = self.fe.extract_features_from_imgs(car_path)
        noncar_feat = self.fe.extract_features_from_imgs(noncar_path)
        features = np.vstack((car_feat, noncar_feat)).astype(np.float64)
        # Normalize feature data
        self.scaler = StandardScaler().fit(features)
        self.features = self.scaler.transform(features)
        self.labels = np.hstack([np.ones(len(car_feat)),
                                 np.zeros(len(noncar_feat))])
        
    def train_and_valid(self):
        '''
        Train loaded dataset using a scikit-learn classification method.
        The loaded dataset are splitted into train and valid parts.
        '''
        # Split dataset
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.features, self.labels, test_size=0.2)
        self.clf.fit(X_train, y_train)
        self.valid_acc = self.clf.score(X_valid, y_valid)
        
    def predict(self, imgs):
        '''Predict images. Input should be a list of image arrays.'''
        features = self.fe.extract_features_from_imgs(imgs, path=False)
        features = self.scaler.transform(features)
        prediction = self.clf.predict(features)
        return prediction
    
    def save_fit(self, fname='saved_fit.p'):
        '''Save fitted classifier and scaler to a pickle file.'''
        saved_dict = {}
        saved_dict['clf'] = self.clf
        saved_dict['scaler'] = self.scaler
        with open(fname, 'wb') as f:
            pickle.dump(saved_dict, f)
        print('Fitted clf and scaler saved to {}.'.format(fname))
    
    def load_fit(self, fname='saved_fit.p'):
        '''Load'''
        with open(fname, 'rb') as f:
            saved_dict = pickle.load(f)
        self.clf = saved_dict['clf']
        self.scaler = saved_dict['scaler']
        print('Fitted clf and scaler loaded from {}.'.format(fname))



assert K.image_dim_ordering() == 'tf', 'Image array should be in tf mode.'

def data_generator(all_paths, all_labels, batch_size=64):
    n_samples = len(all_paths)
    start = 0
    while True:
        end = start + batch_size
        batch_paths = all_paths[start:end]
        batch_x = np.array([plt.imread(path, format='RGB') for path in batch_paths])
        batch_y = all_labels[start:end]
        start += batch_size
        if start >= n_samples:
            start = 0
            all_paths, all_labels = shuffle(all_paths, all_labels)
        batch_x, batch_y = shuffle(batch_x, batch_y)
        yield batch_x, batch_y

def normalize(x):
    '''x should be a tensor'''
    x = K.cast(x, dtype='float32')
    normed = x / 255.0 - 0.5
    return normed

def convnet(features):
    '''features should be an keras Input layer.'''
    normed = Lambda(normalize, name='normed')(features)
    conv1 = Conv2D(16, 3, 3, border_mode='same', activation='relu', name='conv1')(normed)
    pool1 = MaxPooling2D((2, 2), name='pool1')(conv1)
    conv2 = Conv2D(16, 3, 3, border_mode='same', activation='relu', name='conv2')(pool1)
    pool2 = MaxPooling2D((2, 2), name='pool2')(conv2)
    conv3 = Conv2D(16, 3, 3, border_mode='same', activation='relu', name='conv3')(pool2)
    pool3 = MaxPooling2D((2, 2), name='pool3')(conv3)
    flatten = Flatten(name='flatten')(pool3)
    fc1 = Dense(256, activation='relu', name='fc1')(flatten)
    fc1 = Dropout(0.5, name='fc1_dropout')(fc1)
    fc2 = Dense(128, activation='relu', name='fc2')(fc1)
    fc2 = Dropout(0.5, name='fc2_dropout')(fc2)
    predictions = Dense(2, activation='softmax', name='output')(fc2)
    return predictions

class ConvNetClassifier(object):
    
    def __init__(self, config):
        self.config = config
        self.input_shape = (64, 64, 3)
        self._load_data()
        self._add_model()
    
    def _load_data(self):
        all_paths = list(self.config.car_path) + list(self.config.noncar_path)
        all_labels = np.concatenate((np.ones(len(self.config.car_path)),
                                     np.zeros(len(self.config.noncar_path))))
        all_labels = to_categorical(all_labels)
        train_paths, valid_paths, train_labels, valid_labels = train_test_split(
            all_paths, all_labels, test_size=0.2)
        self.train_datagen = data_generator(train_paths, train_labels,
                                            self.config.batch_size)
        self.valid_datagen = data_generator(valid_paths, valid_labels,
                                            self.config.batch_size)
        self.train_n_samples = len(train_paths)
        self.valid_n_samples = len(valid_paths)
    
    def _add_model(self):
        features = Input(shape=self.input_shape, name='features')
        predicts = convnet(features)
        self.model = Model(features, predicts)
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def train(self, fine_tune=False):
        '''Train or fine_tune the model.'''
        weights_path = self.config.save_model_name+'.h5'
        if fine_tune and os.path.exists(weights_path):
            print('Loading {}'.format(weights_path))
            self.model.load_weights(weights_path)
        # callbacks
        save_best_model = ModelCheckpoint(weights_path,
                                          save_best_only=True,
                                          save_weights_only=True)
        callback_list = [save_best_model]
        # train ops
        self.model.fit_generator(self.train_datagen,
                                 samples_per_epoch=self.train_n_samples,
                                 nb_epoch=self.config.epoches,
                                 validation_data=self.valid_datagen,
                                 nb_val_samples=self.valid_n_samples,
                                 callbacks=callback_list)
        
    def predict(self, imgs):
        imgs = np.array(imgs).reshape((-1, 64, 64, 3))
        predicts = self.model.predict(imgs, batch_size=self.config.batch_size)
        return np.argmax(predicts, axis=1)
    
    def load_model(self, weights_path=None):
        if weights_path is None:
            weights_path = self.config.save_model_name+'.h5'
        print('Loading {}'.format(weights_path))
        self.model.load_weights(weights_path)
    
    def evaluate(self, X=None, y=None):
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        if X is None or y is None:
            loss, acc = self.model.evaluate_generator(self.valid_datagen,
                                                      self.valid_n_samples)
        else:
            loss, acc = self.model.evaluate(X, y)
        return acc