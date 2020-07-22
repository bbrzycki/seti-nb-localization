import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout
from keras.layers import Flatten, Input, Dense, BatchNormalization
from keras import backend as K
from keras.utils import plot_model

from sklearn.model_selection import train_test_split

import os, sys, errno
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import pickle
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import time

import argparse


# Global values
PATH = '/datax/scratch/bbrzycki/data/nb-localization'
TCHANS, FCHANS = 32, 1024


def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(TCHANS, FCHANS), n_channels=1, n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        
        # 
        X = np.reshape(np.array([np.load(fn)[:TCHANS, :FCHANS] for fn in list_IDs_temp]),
                       (self.batch_size, *self.dim, 1))
        if self.n_channels == 2:
            X2 = np.copy(X)
        
        # Normalize over entire frame
        X -= np.mean(X, axis=(1, 2, 3), keepdims=True)
        X /= np.std(X, axis=(1, 2, 3), keepdims=True)
        
        if self.n_channels == 2:
            # Normalize over frequency
            X2 -= np.mean(X2, axis=(1, 3), keepdims=True)
            X2 /= np.std(X2, axis=(1, 3), keepdims=True)
            X = np.concatenate([X, X2], axis=3)
        
        # filenames in csv are only the filename, whereas glob dives the rest of the path
        y = np.array([self.labels[os.path.split(fn)[1]] for fn in list_IDs_temp]).reshape((self.batch_size, self.n_classes))

        return X, y

    
def index_diff(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred)**2)**0.5 * FCHANS
    
    
def BaselineModel(n_classes):
    def Conv_Pool(x, num_filters=32):
        x = Conv2D(num_filters, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    inputs = Input(shape=(32, 1024, 1))
    
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    convpool1 = Conv_Pool(conv1, 32)
    convpool2 = Conv_Pool(convpool1, 32)
    convpool3 = Conv_Pool(convpool2, 64)
    
    flat = Flatten()(convpool3)
    dense1 = Dense(64, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    dropout = Dropout(0.5)(dense2)
    
    outputs = Dense(n_classes, activation='linear')(dropout)
    
    model = Model(inputs=inputs, 
                        outputs=outputs)

    model.compile(loss='mean_squared_error',
                  optimizer='adam', 
                  metrics=[index_diff])
    
    return model
    

def FinalModel(n_classes, dense1_num=1024, dense2_num=1024):
    def Residual(x, layers=32):
        conv = Conv2D(layers, (3, 3), padding='same')(x)
        residual =  keras.layers.add([x, conv])
        act = Activation('relu')(residual)
        normed = BatchNormalization()(act)
        return normed

    inputs = Input(shape=(TCHANS, FCHANS, 2))
    
    strided1 = Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
    residual1 = Residual(strided1, 32)
    strided2 = Conv2D(32, (3, 3), strides=2, activation='relu')(residual1)
    residual2 = Residual(strided2, 32)
    strided3 = Conv2D(64, (3, 3), strides=2, activation='relu')(residual2)
    
    flat = Flatten()(strided3)
    dense1 = Dense(dense1_num, activation='relu')(flat)
    dense2 = Dense(dense2_num, activation='relu')(dense1)
    dropout = Dropout(0.5)(dense2)
    
    outputs = Dense(n_classes, activation='linear')(dropout)

    model = Model(inputs=inputs, 
                  outputs=outputs)
    
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=[index_diff])
    
    return model


def choose_data(rfi_num=0, use_bright=False, split_name='train'):
    xsig = '{:d}sig'.format(1 + rfi_num)
    
    # Load dataset paths and labels
    filenames = glob.glob('{}/{}/{}/*.npy'.format(PATH, xsig, split_name))
    
    # For bright case, exclude 0, 5 dB signals
    if use_bright and split_name == 'train':
        dim_filenames = glob.glob('{}/{}/{}/0*.npy'.format(PATH, xsig, split_name))
        filenames = list(set(filenames) - set(dim_filenames))
    training_size = len(filenames)

    labels_df = pd.read_csv('{}/{}/{}/labels.csv'.format(PATH, xsig, split_name))
    if rfi_num == 0:
        labels = {row['filename']: [row['start_index']/FCHANS, 
                                    row['end_index']/FCHANS] 
                  for index, row in labels_df.iterrows()}
    else:
        labels = {row['filename']: [row['start_index']/FCHANS, 
                                    row['end_index']/FCHANS,
                                    row['rfi_start_index']/FCHANS, 
                                    row['rfi_end_index']/FCHANS] 
                  for index, row in labels_df.iterrows()}
        
    return filenames, labels

    
    
def run_training(model_name='baseline', 
                 rfi_num=0,
                 use_bright=False,
                 batch_size=32,
                 epochs=100,
                 seed=42,
                 gpu_id='0',
                 exp_name='',
                 dense1_num=1024,
                 dense2_num=1024):
    # Set which gpu to use per experiment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    start_time = time.time()
    
    xsig = '{:d}sig'.format(1 + rfi_num)
    n_classes = (1 + rfi_num) * 2
    
    # Create directory for models and output files
    model_dir_path = '{}/training/{}_{}_{:d}bs'.format(PATH,
                                                       model_name,
                                                       xsig,
                                                       batch_size)
    if use_bright:
        model_dir_path = '{}_bright'.format(model_dir_path)
    if exp_name != '':
        model_dir_path = '{}_{}'.format(model_dir_path, exp_name)
    mkdir(model_dir_path)
    
    # Set number of channels used in data preprocessing depending on the model
    if model_name == 'baseline':
        n_channels = 1
        model = BaselineModel(n_classes)
    elif model_name == 'final':
        n_channels = 2
        model = FinalModel(n_classes, dense1_num, dense2_num)
    else:
        sys.exit('Invalid model name')
        
    plot_model(model, to_file='{}/architecture.png'.format(model_dir_path), show_shapes=True)
    
    # Load dataset paths and labels
    filenames, labels = choose_data(rfi_num=rfi_num, 
                                    use_bright=use_bright,
                                    split_name='train')

    # Create train/validation split
    X_train, X_validation = train_test_split(filenames, test_size=0.2, random_state=seed)

    # Make dataset generators
    train_params = {'dim': (TCHANS, FCHANS),
                    'batch_size': batch_size,
                    'n_channels': n_channels,
                    'n_classes': n_classes,
                    'shuffle': True}

    
    training_generator = DataGenerator(X_train, labels, **train_params)
    validation_generator = DataGenerator(X_validation, labels, **train_params)
    
    
    # Set up model training using custom generators, with callbacks for early stopping
    model_fn = '{}/model.h5'.format(model_dir_path)
    history_fn = '{}/history'.format(model_dir_path)
    
    history = model.fit(x=training_generator,
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(X_validation) // batch_size,
                        use_multiprocessing=True,
                        workers=8,
                        callbacks=[keras.callbacks.ModelCheckpoint(model_fn,
                                                                   monitor='val_loss',
                                                                   verbose=0, 
                                                                   save_best_only=True,
                                                                   mode='auto'), 
                                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                     factor=0.1, 
                                                                     patience=5, 
                                                                     min_lr=1e-6), 
                                   keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                 patience=10,
                                                                 verbose=0, 
                                                                 mode='auto')])

    # Save models and history
    model.save_weights(model_fn)
    
    time_elapsed = time.time() - start_time
    history.history['time_elapsed'] = time_elapsed
    history.history['gpu_id'] = gpu_id
    with open(history_fn, 'wb') as f:
        pickle.dump(history.history, f)
        
    print('Training time: {:.2f} min'.format(time_elapsed/60))
    
    
    
    # Make predictions on test data
    test_filenames, test_labels = choose_data(rfi_num=rfi_num, 
                                              use_bright=use_bright,
                                              split_name='test')
    
    test_params = {'dim': (TCHANS, FCHANS),
                   'batch_size': 32, # ensure # test divisible by batch size
                   'n_channels': n_channels,
                   'n_classes': n_classes,
                   'shuffle': False}
    
    test_generator = DataGenerator(test_filenames, test_labels, **test_params)
    predictions = model.predict(x=test_generator)
    
    true_labels = [test_labels[os.path.split(fn)[1]] for fn in test_filenames]
    pred_dict = {os.path.split(fn)[1]: np.stack([tr, pr]) 
                 for fn, tr, pr in zip(test_filenames, true_labels, predictions)}
    
    np.save('{}/test_predictions.npy'.format(model_dir_path), pred_dict)
    
    
def main():
    training_path = '{}/training'.format(PATH)
    mkdir(training_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default='baseline', required=True)
    parser.add_argument('--rfi_num', '-r', type=int, default=0) # 0 or 1
    parser.add_argument('--use_bright', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--dense1_num', type=int, default=1024)
    parser.add_argument('--dense2_num', type=int, default=1024)
    args = parser.parse_args()
    
    # Convert args to dictionary
    params = vars(args)
    
    run_training(**params)

        
if __name__ == "__main__":
    main()