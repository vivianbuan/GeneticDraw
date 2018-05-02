import numpy as np
import os
import tensorflow as tf
from TFUtils import TFUtils
from tensorflow.examples.tutorials.mnist import input_data


def load_training_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = script_dir + '/mnist/data'

    print('Preparing MNIST data..')

    mnist = input_data.read_data_sets(data_path, one_hot=True)

    print(np.array(mnist).shape)
    print(mnist)

    X = mnist.test.images.reshape(-1, 28, 28, 1)
    Y = mnist.test.labels
    X = mnist.train.images.reshape(-1, 28, 28, 1)

    own_X = load_own_data()
    X = np.vstack((X, own_X))
    Y = add_unknown(Y, len(own_X))


def add_own_data(X, Y, dataset='train'):
    own_X = load_own_data(dataset=dataset)
    X = np.vstack((X, own_X))
    Y = add_unknown(Y, len(own_X))
    indices = np.random.choice(len(Y), len(Y), replace=False)
    X = X[indices]
    Y = Y[indices]
    return X, Y


def add_unknown(Y, num):
    Y = np.hstack((Y, np.zeros((Y.shape[0], 1))))
    one_hot = np.zeros((num, Y.shape[1]))
    one_hot[:, -1] = 1
    Y = np.vstack((Y, one_hot))
    return Y


# Delete entries with labels be drop_out
# Delete columns of the drop_out label
def drop_out_data(X, Y, drop_out):
    labels = np.argmax(Y, axis=1)
    indices = np.where(labels != drop_out)[0]
    Y = Y[indices, :]
    X = X[indices, :]

    Y = delete_drop_out_label(Y, drop_out)

    return X, Y


def delete_drop_out_label(Y, drop_out):
    return np.delete(Y, drop_out, 1)


def load_own_data(dataset=r'train'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = script_dir + '/unknown_images/'

    X = np.load(os.path.join(data_path, 'data_' + dataset + '.npy'))
    return X


def get_model_path(model_path, i_test):

    if i_test == 11:
        has_unknown = True
        drop_out = None
    elif i_test == 10:
        has_unknown = False
        drop_out = None
    else:
        has_unknown = True
        drop_out = i_test

    if has_unknown:
        model_path += '/has-unknown'
        if drop_out is None:
            model_path += '/mnist-cnn'
    else:
        model_path += '/ori'

    if drop_out is not None:
        model_path += '/drop-out/' + str(drop_out)

    return model_path, has_unknown, drop_out


if __name__ == '__main__':
    load_training_data()
    # Y = np.zeros((3,10))
    # Y[0, 1] = 1
    # Y[1, 3] = 1
    # Y[2, 2] = 1


    # load_own_data()
