import warnings
warnings.simplefilter('ignore')

import numpy as np
from energyflow.utils import to_categorical
from energyflow.archs import EFN
import os.path as osp, os
from time import strftime
from tensorflow.keras.callbacks import ModelCheckpoint


def load(npz):
    d = np.load(npz)
    return d['X'], d['y'], d['inpz']


def main():
    X_train, y_train, inpz_train = load('data/train/merged.npz')
    X_test, y_test, inpz_test = load('data/test/merged.npz')

    # One-hot encoded
    Y_train = to_categorical(y_train, 2)
    Y_test = to_categorical(y_test, 2)

    Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
    efn = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes)

    efn.load_weights('ckpts_Apr06_134930/ckpt-40-val_acc-0.73.hdf5')

    ckpt_dir = strftime('ckpts_%b%d_%H%M%S')
    if not osp.isdir(ckpt_dir): os.makedirs(ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        ckpt_dir + '/ckpt-{epoch:02d}-val_acc-{val_acc:.3f}.hdf5',
        monitor='val_acc', verbose=1, save_best_only=False, mode='max'
        )
    best_checkpoint_callback = ModelCheckpoint(
        ckpt_dir + '/ckpt-best.hdf5',
        monitor='val_acc', verbose=1, save_best_only=True, mode='max'
        )

    efn.fit(
        [X_train[:,:,0], X_train[:,:,1:]], Y_train,
        epochs=40,
        batch_size=64,
        validation_data=([X_test[:,:,0], X_test[:,:,1:]], Y_test),
        verbose=1,
        callbacks=[checkpoint_callback, best_checkpoint_callback]
        )


if __name__ == '__main__':
    main()