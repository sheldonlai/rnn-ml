import pandas as pd
import tensorflow as tf

def run_eval(model_path, eval_set):
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
    # target = tf.placeholder(tf.float32, [None, output_dim])

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

    # Define optimizer and training operations
    with tf.name_scope('gloabl_step'):
        global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate)

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
                raise Exception("Couldn't load model")

        result = np.zeros(1, len(eval_set))
        # model should be loaded
        eval_step = 0
        while True:
            test_x, test_y = get_test_data(eval_step)
            if test_x is None:
                break
            eval_dict = {
                input: test_x
            }
            results = sess.run([prediction], feed_dict=eval_dict)
            print (results)
            eval_step += 1





if __name__ == '__main__':
    run_eval('soem _ path', pd.read_csv("/final.csv"))
