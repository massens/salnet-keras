'''
	Model: model1_salnet_keras
'''

from keras.optimizers import SGD
import h5py
import numpy as np
import keras
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation
from keras.layers import Activation, BatchNormalization, Input
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import pandas as pd
keras.backend.set_image_dim_ordering("th")

IN_DIR='/root/sharedfolder/salnet_keras/input/'
OUT_DIR ='output/model1_salnet_keras_10_smaller_lr/'


class PeriodicLR(object):
    """
    Learning rate schedule that periodically reduces
    """
    def __init__(self, base_lr, epochs, gamma):
        self.base_lr = base_lr
        self.epochs = epochs
        self.gamma = gamma

    def __call__(self, epoch):
        n = epoch / self.epochs
        return self.base_lr * (self.gamma ** n)


def load_datasets():
    DATASET_FILE = '/root/sharedfolder/salnet_keras/input/dataset_for_vgg16keras_normalized.hdf5'
    f = h5py.File(DATASET_FILE)
    X_train = f['train/stimuli']
    Y_train = f['train/saliency']
    X_val = f['val/stimuli']
    Y_val = f['val/saliency']
    print X_train.shape, Y_train.shape
    print X_val.shape, Y_val.shape
    return (X_train, Y_train), (X_val, Y_val)


def norm_weights(n):
    r = n / 2.0
    xs = np.linspace(-r, r, n)
    x, y = np.meshgrid(xs, xs)
    w = np.exp(-0.5*(x**2 + y**2))
    w /= w.sum()
    return w

def deconv(nb_filter, size, name):
    upsample = UpSampling2D(size=(size, size))
    s = 2 * size + 1
    w = norm_weights(s)[np.newaxis, np.newaxis, :, :]
    conv = Convolution2D(
        nb_filter, s, s,
        name=name,
        activation='linear',
        bias=False,
        border_mode='same',
        weights=[w])
    return lambda x: conv(upsample(x))

def get_model():

    input_tensor = Input(shape=(3, 240, 320)) 
    base_model = VGG16(weights='imagenet', input_tensor=input_tensor)

    x = base_model.get_layer('block3_conv3').output
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv4')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv5')(x)
    x = Convolution2D(256, 7, 7, activation='relu', border_mode='same', name='conv6')(x)
    x = Convolution2D(128, 11, 11, activation='relu', border_mode='same', name='conv7')(x)
    x = Convolution2D(32 , 11, 11, activation='relu', border_mode='same', name='conv8')(x)
    x = Convolution2D(1 , 13, 13, activation='relu', border_mode='same', name='conv9')(x)
    x = UpSampling2D(size=(4, 4))(x)
    x = Convolution2D(1, 9, 9, name='deconv', bias=False, border_mode='same', activation='linear')(x)
    # x = deconv(1,4, 'deconv')(x)
    output = Activation('sigmoid')(x)

    model = Model(input=input_tensor, output=output)

    # for layer in base_model.layers:
    #     w = layer.get_weights()
    #     if len(w) > 0 : # Is convolutional
    #         w[1] = w[1] * (1.0/150)
    #         layer.set_weights(w)

    sgd = SGD(lr=1.3e-7)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model


def main():
    model = get_model()
    # model.load_weights('/root/sharedfolder/salnet_keras/output/model1_salnet_keras_4/model.08.weights')

    model.summary()
    # model.load_weights('/root/sharedfolder/salnet_keras/output/model1_8/model.16-0.30858.weights')
    
    (X_train, Y_train), val = load_datasets()


    checkpointer = ModelCheckpoint(
        OUT_DIR + 'model.{epoch:02d}-{val_loss:.5f}.weights',
        monitor='val_loss',
        save_best_only=False)

    lr = 0.001
    lr_scheduler = LearningRateScheduler(
        schedule=PeriodicLR(lr, 5, 0.5))

    try:
        model.fit(
            X_train,
            Y_train,
            batch_size=2 ,
            nb_epoch=90,
            shuffle="batch",
            validation_data = val,
            callbacks=[checkpointer])

    finally:

        if hasattr(model, 'history'):
            history = pd.DataFrame(model.history.history)
            history.to_csv(OUT_DIR + 'train_salvol20.log', index_label='epoch')
            print(history)

        try:
            model.save_weights(OUT_DIR + 'final.weights')
            print('Saved final weights in snapshots/salvol20/final.weights')
        except Exception as e:
            print(e)
            ############# Results


if __name__ == '__main__':
    main()