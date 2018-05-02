import os

from MNISTTrainer import MNISTTrainer
import tensorflow as tf
import debug
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
####################
has_unknown = True
drop_out = 2

# directory settings
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = script_dir + '/mnist/data/'

for i_train in range(12):
    '''
    if i_train == 10:
        has_unknown = False
        drop_out = None
    else:
        has_unknown = True
        drop_out = i_train
    '''

    model_path = script_dir + '/models'
    log_path = script_dir + '/logs'

    ''''
    if has_unknown:
        model_path += '/has-unknown'
        log_path += '/has-unknown'
    else:
        model_path += '/ori'
        log_path += '/ori'

    if drop_out is not None:
        model_path += '/drop-out/' + str(drop_out)
        log_path += '/drop-out/' + str(drop_out)
    '''

    path, has_unknown, drop_out= debug.get_model_path('', i_train)

    model_path += path
    log_path += path

    print(i_train, ': ', has_unknown, drop_out)
    print('model_path', model_path)
    print('log_path', log_path)
    ##########

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        with tf.variable_scope('mnist-cnn' + str(i_train)):
            mnist = MNISTTrainer(
                        data_path=data_path,
                        model_path=model_path,
                        log_path=log_path,
                        has_unknown=has_unknown,
                        drop_out=drop_out)

            mnist.training(
                learning_rate=0.001,
                decay=0.9,
                training_epochs=10,
                batch_size=100,
                p_keep_conv=0.8,
                p_keep_hidden=0.5)
