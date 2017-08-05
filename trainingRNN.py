import datetime
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn

from simple_converter import get_train_test_sets


def run_training(train_df, test_df, model_dir=None):
    model_path=None
    summaries_dir = ''

    if model_dir is not None:
        with open(os.path.join(model_dir, "checkpoint"), "rU") as cp:
            match = re.match('model_checkpoint_path: \"(.+)\"', cp.readline())
            model_path = os.path.join(model_dir, match.group(1))
            summaries_dir = model_dir.replace("models", "log")
    else:
        date_str = datetime.datetime.now().strftime("%I-%M-%p-%B-%d-%Y")
        summaries_dir = './log/%s' % date_str
        model_dir = './models/%s' % date_str
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    model_save_path = os.path.join(model_dir, "model.ckpt")

    # splited data
    df_target = train_df['logerror']
    df_features = train_df.drop(['logerror'], axis=1, inplace=False)
    # constants
    test_df_target = test_df['logerror']
    test_df_features = test_df.drop(['logerror'], axis=1, inplace=False)

    learning_rate = 0.001
    batch_size = 1028
    hidden_units_size = 1028

    input_dim = len(df_features.columns)
    output_dim = 1
    regularize_rate = 0.2



    # x, y placeholder
    input = tf.placeholder(tf.float32, [None, batch_size, input_dim])
    target = tf.placeholder(tf.float32, [None, output_dim])

    # create RNN
    lstm = rnn.MultiRNNCell([
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
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(target, prediction))
        regularize = tf.nn.l2_loss(w)
        loss = tf.reduce_mean(loss + regularize_rate * regularize)
        tf.summary.scalar('loss', loss)

    # Define optimizer and training operations
    with tf.name_scope('gloabl_step'):
        global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.name_scope('train'):
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.sqrt(loss), tf.float32))
        tf.summary.scalar('acc', accuracy)

    def get_input_data(i):
        get_list = [n % (len(train_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = train_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

    def get_test_data(i):
        if i * batch_size > len(test_df):
            return None, None
        get_list = [n % (len(test_df) - 1) for n in range(batch_size * i, batch_size * i + batch_size)]
        samples = test_df.iloc[get_list]
        target = samples.pop('logerror').values.reshape(batch_size, 1)
        features = samples.values
        features = features.reshape(1, batch_size, input_dim)
        return features, target

        # Evaluation

    def evaluation(sess, merged, writer):
        test_step = 0
        mean = 0
        while True:
            test_x, test_y = get_test_data(test_step)
            if test_x is None:
                break
            eval_dict = {
                input: test_x,
                target: test_y
            }
            summary, eval_loss = sess.run([merged, accuracy], feed_dict=eval_dict)

            mean += eval_loss
            test_step += 1

        mean /= (test_step + 1)
        print("Mean test loss is %f" % mean)
        test_summ = tf.Summary()
        test_summ.value.add(tag='test_accuracy',
                             simple_value=mean)
        writer.add_summary(test_summ, i)
        writer.add_summary(summary, i)
        return summary, mean


    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        timer = tf.train.SecondOrStepTimer(every_steps=10)
        sess.run(tf.global_variables_initializer())
        if model_path:
            try:
                saver.restore(sess, model_path)
                print("retrieved model:")
            except Exception as e:
                print("unable to retrieve model\n  %s", e.message)

        for i in range(1, 7001):
            input_data, output_data = get_input_data(i)

            feed = {
                input: input_data,
                target: output_data
            }

            # This runs your optimizer one step
            sess.run(train_op, feed_dict=feed)

            if timer.should_trigger_for_step(i):
                elapsed_time, elapsed_steps = timer.update_last_triggered_step(i)
                if (i != 1):
                    print("%d steps used %f seconds" % (elapsed_steps, elapsed_time))

            if i % 50 == 0:
                step = tf.train.global_step(sess, global_step)
                summary, acc = sess.run([merged, loss], feed_dict=feed)
                print('At step %d loss is: %f' % (step, np.sqrt(acc)))
                train_summ = tf.Summary()
                train_summ.value.add(tag='train_accuracy',
                                     simple_value=np.sqrt(acc))
                train_writer.add_summary(train_summ, i)
                train_writer.add_summary(summary, i)

            if i % 100 == 0:
                evaluation(sess, merged, train_writer)
                save_path = saver.save(sess, model_save_path, global_step=global_step)
                print('results saved to: %s' % save_path)
        evaluation(sess, merged, train_writer)


if __name__ == '__main__':
    try:
        test = pd.read_csv('./testing.csv', dtype='float32')
        train = pd.read_csv('./training.csv', dtype='float32')
    except:
        train, test = get_train_test_sets(0.8)
    run_training(train, test)
