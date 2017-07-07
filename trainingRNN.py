import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from simple_converter import get_train_test_sets


def run_training(train_df, test_df):
    # splited data
    df_target = train_df['logerror']
    df_features = train_df.drop(['logerror'], axis=1, inplace=False)
    # constants

    learning_rate = 0.0001
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
        rnn.BasicLSTMCell(hidden_units_size)
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

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_input_data(i):
        get_list = [n % (len(train_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = train_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

    def get_test_data(i):
        if i >  len(test_df):
            return None, None;
        get_list = [n % (len(test_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = test_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, './RNN-data/my-model')
        except:
            sess.run(tf.global_variables_initializer())
        # Doing 1000 iterations here, feel free to change this
        for i in range(1, 1001):
            input_data, output_data = get_input_data(i)

            feed = {
                input: input_data,
                target: output_data
            }

            # This runs your optimizer one step
            sess.run(train_op, feed_dict=feed)

            if i % 10 == 0:
                # This will print your training loss every 100 iterations

                print('global step: %d' % global_step)
                print('At stop %d loss is: %f' % (i, sess.run(loss, feed_dict=feed)))
            if i % 1000 == 0:
                saver.save(sess, './RNN-data/my-model', global_step=global_step)
        while True:
            input_data, output_data = get_input_data(i)
            if input_data is None:
                break;
            feed = {
                input: input_data,
                target: output_data
            }



if __name__ == '__main__':
    train, test = get_train_test_sets(0.8)
    run_training(train, test)
