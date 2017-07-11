import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from os import path

from simple_converter import get_train_test_sets


def run_training(train_df, test_df):
    # splited data
    df_target = train_df['logerror']
    df_features = train_df.drop(['logerror'], axis=1, inplace=False)
    # constants
    test_df_target = test_df['logerror']
    test_df_features = test_df.drop(['logerror'], axis=1, inplace=False)

    model_path = './tmp/model.ckpt';

    learning_rate = 0.00001
    batch_size = 128
    hidden_units_size = 1024

    input_dim = len(df_features.columns)
    output_dim = 1

    # x, y placeholder
    input = tf.placeholder(tf.float32, [None, batch_size, input_dim])
    target = tf.placeholder(tf.float32, [None, output_dim])

    # create RNN
    lstm = rnn.MultiRNNCell([
        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),

        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),
        rnn.BasicLSTMCell(hidden_units_size),
    ])
    out, state = tf.nn.dynamic_rnn(lstm, input, dtype=tf.float32)

    # Reshape output of RNN
    out = tf.transpose(out, [1, 0, 2])
    out = tf.gather(out, int(out.get_shape()[0]) - 1)

    # Create Linear layer at the end of RNN
    w = tf.get_variable("weights", [hidden_units_size, output_dim])
    b = tf.get_variable("biases", [output_dim])
    prediction = tf.matmul(out, w) + b

    # Define loss function for the network
    loss = tf.losses.mean_squared_error(target, prediction)

    # Define optimizer and training operations
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(tf.sqrt(loss), tf.float32))

    def get_input_data(i):
        get_list = [n % (len(train_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = train_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

    def get_test_data(i):
        if i * batch_size > len(test_df):
            return None, None;
        get_list = [n % (len(test_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = test_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, './tmp/model.ckpt-6500')
            print("retrieved model:")
        except Exception as e:
            print("unable to retrieve model\n  %s", e.message)

        for i in range(1, 2001):
            input_data, output_data = get_input_data(i)

            feed = {
                input: input_data,
                target: output_data
            }

            # This runs your optimizer one step
            sess.run(train_op, feed_dict=feed)

            if i % 100 == 0:
                step = tf.train.global_step(sess, global_step)
                print('At stop %d loss is: %f' % (step, np.sqrt(sess.run(loss, feed_dict=feed))))
            if i % 500 == 0:
                save_path = saver.save(sess, model_path, global_step=global_step)
                print('results saved to: %s' % save_path)

        # Evaluation
        i = 0
        mean = 0;
        while True:
            test_x, test_y = get_test_data(i)
            if test_x is None:
                break
            eval_dict = {
                input: test_x,
                target: test_y
            }
            eval_loss = sess.run(accuracy, feed_dict=eval_dict)
            print("At step %d loss is %f" % (i, eval_loss))
            mean += eval_loss
            i += 1

        mean /= (i + 1)
        print(mean)


if __name__ == '__main__':
    try:
        test = pd.read_csv('./testing.csv', dtype='float32')
        train = pd.read_csv('./training.csv', dtype='float32')
    except:
        train, test = get_train_test_sets(0.8)
    run_training(train, test)
