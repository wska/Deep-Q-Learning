import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2


def default_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


def twoLayerModel(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform', kernel_regularizer=l2(self.regularization)))

        model.add(Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform', kernel_regularizer=l2(self.regularization)))

        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


def SGDtwoLayerModel(self):
        sgd = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform', kernel_regularizer=l2(self.regularization)))

        model.add(Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform', kernel_regularizer=l2(self.regularization)))
                        
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=sgd)
        return model
