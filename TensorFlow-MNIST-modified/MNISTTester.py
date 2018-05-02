import os
from random import randint

import numpy as np
import tensorflow as tf
from MNIST import MNIST
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import debug
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# MNIST Tester class
# check accuracy of test set
# predict random number from test set
# predict number from image

#has_unknown = True
#drop_out = 1


class MNISTTester(MNIST):
    def __init__(self, model_path=None, data_path=None, has_unknown=False, drop_out=None):
        MNIST.__init__(self, model_path, data_path, has_unknown, drop_out)
        self.has_unkown = has_unknown
        self.drop_out = drop_out
        self.init()

    def init(self):
        self.print_status('Loading a model..')
        self.init_session()
        self.load_model()

        if self.data_path is not None:
            self.load_training_data(self.data_path)

        self.prediction = tf.nn.softmax(self.model)

    def classify(self, feed_dict):
        number = self.sess.run(tf.argmax(self.model, 1), feed_dict)[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict)[0]
        accuracy = self.add_drop_out_acc(accuracy)
        number = self.switch_drop_out_number(number)

        return number, accuracy[number]

    def classify_all(self, feed_dict):
        # accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict)[0]
        accuracy = self.sess.run(self.prediction, feed_dict)[0]
        accuracy = self.add_drop_out_acc(accuracy)
        number = np.argsort(accuracy)[::-1]

        # return number, accuracy[number]
        return number, accuracy

    def accuracy_of_testset(self):
        self.print_status('Calculating accuracy of test set..')

        X = self.mnist.test.images.reshape(-1, 28, 28, 1)
        Y = self.mnist.test.labels
        test_feed_dict = self.build_feed_dict(X, Y)
        accuracy = self.check_accuracy(test_feed_dict)
        accuracy = self.add_drop_out_acc(accuracy)

        self.print_status('CNN accuracy of test set: %f' % accuracy)

    def predict_random(self, show_image=False):
        num = randint(0, self.mnist.test.images.shape[0])
        image = self.mnist.test.images[num]
        label = self.mnist.test.labels[num]

        feed_dict = self.build_feed_dict(image.reshape(-1, 28, 28, 1), [label])

        (number, accuracy) = self.classify(feed_dict)
        label = self.sess.run(tf.argmax(label, 0))

        self.print_status('Predict random item: %d is %d, accuracy: %f' %
                          (label, number, accuracy))

        if show_image is True:
            plt.imshow(image.reshape(28, 28))
            plt.show()

    def predict(self, filename):
        data = self.load_image(filename)

        number, accuracy = self.classify({self.X: data})

        self.print_status('%d is %s, accuracy: %f' % (number, os.path.basename(filename), accuracy))

    def predict_img(self, data, digit=None):

        _, accuracy = self.classify_all({self.X: data})

        try:
            return accuracy[digit]
        except:
            return accuracy
        '''
        for i in range(len(number)):
            n = number[i]
            self.print_status('%d is %s, accuracy: %f' % (n, os.path.basename(filename), accuracy[n]))
        '''

    def add_drop_out_acc(self, accuracy):
        if self.drop_out is not None:
            accuracy = np.insert(accuracy, self.drop_out, [0])
        if not self.has_unkown:
            accuracy = np.append(accuracy, [0])
        return accuracy

    def switch_drop_out_number(self, number):
        if self.drop_out is not None:
            if self.drop_out <= number:
                return number + 1
            else:
                return number
        else:
            return number

    def load_image(self, filename):
        img = Image.open(filename).convert('L')

        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()

    def convert_img(self, img):
        img = img.convert('L')

        # resize to 28x28
        #img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        img = img.resize((28, 28), Image.ANTIALIAS)

        #print(list(img.getdata()))
        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()


def convert_img(img):
    img = img.convert('L')

    # resize to 28x28
    #img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    img = img.resize((28, 28), Image.ANTIALIAS)

    #print(list(img.getdata()))
    # normalization : 255 RGB -> 0, 1
    data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

    # reshape -> [-1, 28, 28, 1]
    return np.reshape(data, (-1, 28, 28, 1)).tolist()


def preload(num_models=12):
    global mnist_models, flags
    mnist_models = []
    flags = []
    ####################
    # directory settings
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = script_dir + '/mnist/data/'
    model_path = script_dir + '/models'

    for i_test in range(0, num_models):

        data_path = script_dir + '/mnist/data/'
        model_path = script_dir + '/models'

        model_path, has_unknown, drop_out = debug.get_model_path(model_path, i_test)
        print(i_test, 'model_path', model_path)
        #####################################
        # prediction test with MNIST test set
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            with tf.variable_scope('mnist-cnn' + str(i_test)):
                mnist = MNISTTester(
                    model_path=model_path,
                    data_path=data_path,
                    has_unknown=has_unknown,
                    drop_out=drop_out)
                mnist_models.append(mnist)
                flags.append(i_test)


def predict(img, label=0):
    global mnist_models, flags
    if img is None:
        if label is None:
            return np.array([float('nan')] * 11)
        else:
            return float('nan')
    img = convert_img(img)

    probs = np.zeros(11)
    counts = np.zeros(probs.shape)
    indices = np.arange(0, len(probs))
    for i_test, (mnist, flag) in enumerate(zip(mnist_models, flags)):
        this_acc = mnist.predict_img(img, digit=None)
        this_acc = this_acc.flatten()
        probs += this_acc
        counts[indices != flag] += 1
    acc = probs / counts
    acc /= sum(acc)
    if label is None:
        return acc
    else:
        # l2 norm distance
        # target = np.zeros(acc.size)
        # target[label] = 1
        # return -np.sum((target - acc) ** 2)
        target = np.ones(acc.size) * 1e-10
        target[label] = 1
        acc[acc == 0] = 1e-10
        value = -np.sum(acc * np.log(acc/target))
    return value


def multi_predict(path=None):
    preload()
    import pandas as pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        mypath = script_dir + '/test_images'
    else:
        mypath = path

    probabilities = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for _, filename in enumerate(onlyfiles):
        try:
            img = Image.open(os.path.join(mypath, filename)).convert('L')
        except:
            img = None
        prob = predict(img, label=None)
        probabilities.append(prob)
    probabilities = np.stack(probabilities)
    class_names = np.arange(0, probabilities.shape[1])
    df = pd.DataFrame(data=probabilities, columns=class_names)
    df.insert(loc=0, column='img_name', value=onlyfiles)
    df.to_csv(mypath + '\\' + 'test.csv')


def multi_folder_predict(paths=[]):
    preload()
    import pandas as pd

    for mypath in paths:
        probabilities = []
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for _, filename in enumerate(onlyfiles):
            try:
                img = Image.open(os.path.join(mypath, filename)).convert('L')
            except:
                img = None
            prob = predict(img, label=None)
            probabilities.append(prob)
        probabilities = np.stack(probabilities)
        class_names = np.arange(0, probabilities.shape[1])
        df = pd.DataFrame(data=probabilities, columns=class_names)
        df.insert(loc=0, column='img_name', value=onlyfiles)
        df.to_csv(mypath + '\\' + 'test_forest.csv')


def predict_raw(img):
    global mnist_models, flags
    if img is None:
        return np.ones((len(mnist_models), 11)) * float('nan')
    img = convert_img(img)

    probs = np.zeros((len(mnist_models), 11))
    for i_test, (mnist, flag) in enumerate(zip(mnist_models, flags)):
        this_acc = mnist.predict_img(img, digit=None)
        this_acc = this_acc.flatten()
        probs[i_test] = this_acc
    return probs


def multi_predict_raw(path=None):
    preload()
    import pandas as pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        mypath = script_dir + '/test_images'
    else:
        mypath = path

    probabilities = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    column_names = []

    for _, filename in enumerate(onlyfiles):
        try:
            img = Image.open(os.path.join(mypath, filename)).convert('L')
        except:
            img = None
        prob = predict_raw(img)
        probabilities.append(prob)
        column_names.append([filename + "{0}".format(i) for i in range(12)])
    probabilities = np.vstack(probabilities)
    column_names = np.array(column_names).flatten()
    class_names = np.arange(0, probabilities.shape[1])
    df = pd.DataFrame(data=probabilities, columns=class_names)
    df.insert(loc=0, column='img_name', value=column_names)
    df.to_csv(mypath + '\\' + 'test_raw.csv')


if __name__ == '__main__':
    multi_predict()
    #multi_predict_raw()
    #multi_folder_predict(
    #    paths=[r'D:\360Sync\OneDrive\Berkeley\MachineLearning\Spring2018\Project\Figures\Figure5\reference'])
    #preload()
