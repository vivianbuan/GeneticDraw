import os
import numpy as np

from MNISTTester import MNISTTester

####################
# directory settings
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = script_dir + '/mnist/data/'
model_path = script_dir + '/models'

#####################################
# Specify path for different model
has_unknown = True
drop_out = 1

if has_unknown or drop_out > 0:
    model_path += '/has-unknown'
else:
    model_path += '/ori'

if drop_out > 0:
    model_path += '/drop-out/' + str(drop_out)



#####################################
# prediction test with MNIST test set
# THIS
mnist = MNISTTester(
            model_path=model_path,
            data_path=data_path,
            has_unknown=has_unknown,
            drop_out=drop_out)

# mnist.accuracy_of_testset()
# mnist.predict_random()

#################################
# prediction test with image file
# mnist = MNISTTester(model_path)
mnist.predict(script_dir + '/imgs/digit-1.png')
mnist.predict(script_dir + '/imgs/digit-2.png')
mnist.predict(script_dir + '/imgs/digit-4.png')
mnist.predict(script_dir + '/imgs/digit-5.png')

# return all probabilities
# mnist.predict_all(script_dir + '/imgs/5_3.jpeg')

# return probability of a digit
# mnist.predict_digit(script_dir + '/imgs/test0.jpeg', 5)

# mnist.predict(script_dir + '/imgs/test0.png')
# mnist.predict(script_dir + '/imgs/test_10000.png')
# mnist.predict(script_dir + '/imgs/gen_5000.png')

# def preload():
#     global mnist
#     ####################
#     # directory settings
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#
#     data_path = script_dir + '/mnist/data/'
#     model_path = script_dir + '/models'
#
#     if has_unknown:
#         model_path += '/has-unknown'
#     else:
#         model_path += '/ori'
#
#     if drop_out > 0:
#         model_path += '/drop-out/' + str(drop_out)
#
#
#     #####################################
#     # prediction test with MNIST test set
#     mnist = MNISTTester(
#         model_path=model_path,
#         data_path=data_path,
#         has_unknown=has_unknown,
#         drop_out=drop_out)
#
#
# def predict(img, digit=5):
#     return mnist.predict_img(img, digit)[1]
