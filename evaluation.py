import tensorflow as tf

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from os import path

from simple_converter import get_train_test_sets


def run_eval(test):

    df_target = test['logerror']
    df_features = test.drop(['logerror'], axis=1, inplace=False)

    learning_rate = 0.00001
    batch_size = 128
    hidden_units_size = 1024

    input_dim = len(df_features.columns)
    output_dim = 1

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, './tmp/model.ckpt-6500')
            print("retrieved model:")
        except Exception as e:
            print("unable to retrieve model\n  %s", e.message)

        for i in range(1, 2001):
            input = tf.placeholder(tf.float32, [None, batch_size, input_dim])
            target = tf.placeholder(tf.float32, [None, output_dim])
            feed = {
                input:  df_features,
                target: df_target
            }

            # This runs your optimizer one step
            sess.run(train_op, feed_dict=feed)

            if i % 100 == 0:
                step = tf.train.global_step(sess, global_step)
                print('At stop %d loss is: %f' % (step, np.sqrt(sess.run(loss, feed_dict=feed))))
            if i % 500 == 0:
                save_path = saver.save(sess, model_path, global_step=global_step)
                print('results saved to: %s' % save_path)
        # while True:
        #     input_data, output_data = get_input_data(i)
        #     if input_data is None:
        #         break;
        #     feed = {
        #         input: input_data,
        #         target: output_data
        #     }

if __name__ == '__main__':
    test = pd.read_csv('./testing.csv', dtype='float32')
    run_eval(test)
