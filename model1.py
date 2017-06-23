'''
    Model: model1
'''
from keras.optimizers import SGD
import h5py
import numpy as np
import keras
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import pandas as pd
keras.backend.set_image_dim_ordering("th")

IN_DIR='/root/sharedfolder/salnet_keras/input/'
OUT_DIR ='output/model1_5/'


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
    DATASET_FILE = IN_DIR + 'dataset_salnet_keras_small.hdf5'
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

    x = base_model.get_layer('block2_pool').output
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv3')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv4')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv5')(x)
    x = Convolution2D(256, 7, 7, activation='relu', border_mode='same', name='conv6')(x)
    x = Convolution2D(128, 11, 11, activation='relu', border_mode='same', name='conv7')(x)
    x = Convolution2D(32 , 11, 11, activation='relu', border_mode='same', name='conv8')(x)
    output = Convolution2D(1  , 11, 13, activation='sigmoid', border_mode='same', name='conv9')(x)
   # x = deconv(1,4, 'deconv')(x)
    #out = Activation('sigmoid')(x)

    model = Model(input=input_tensor, output=output )


    sgd = SGD(lr=1e-3, decay=0.005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model


def main():
    model = get_model()
    model.summary()
    (X_train, Y_train), val = load_datasets()

    lr = 0.001
    lr_scheduler = LearningRateScheduler(
        schedule=PeriodicLR(lr, 5, 0.5))

    checkpointer = ModelCheckpoint(
        OUT_DIR + 'model.{epoch:02d}.weights',
        monitor='val_loss',
        save_best_only=False)

    try:
        model.fit(
            X_train[:100],
            Y_train[:100],
            batch_size=10,
            nb_epoch=1,
            shuffle=False,
            validation_split=.1,
            callbacks=[checkpointer, lr_scheduler])

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