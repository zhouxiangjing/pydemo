#coding:utf-8
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
from datetime import datetime
import tensorflow as tf
from captcha_datasets import *

CAPTCHA_TRAIN_MOEDL_DIR = "./captcha_models/"
CAPTCHA_BOARD_LOG_DIR = "./logs/"

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))
    #out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32])) # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    #out = tf.nn.softmax(out)
    return out

# 训练
def captcha_train_cnn():

    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar("loss", loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_MAX_CAPTCHA, CAPTCHA_CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_MAX_CAPTCHA, CAPTCHA_CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    # 加载数据集
    train_tfrecord_files = [CAPTCHA_TRAIN_TFRECORD + "train_captcha_" + str(i) + ".tfrecord" for i in
                            range(CAPTCHA_TRAINT_TFRECORD_COUNT)]
    test_tfrecord_files = [CAPTCHA_TEST_TFRECORD + "test_captcha_" + str(i) + ".tfrecord" for i in
                           range(CAPTCHA_TEST_TFRECORD_COUNT)]
    batch_train_x, batch_train_y = get_dataset(train_tfrecord_files)
    batch_test_x, batch_test_y = get_dataset(test_tfrecord_files)
    print("batch_train_x :", batch_train_x)
    print("batch_train_x :", batch_train_y)
    print("batch_test_x :", batch_test_x)
    print("batch_test_y :", batch_test_y)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    saver = tf.train.Saver(max_to_keep=3)
    #with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        # TensorBoard open log with "tensorboard --logdir=logs"
        if os.path.exists(CAPTCHA_BOARD_LOG_DIR):
            shutil.rmtree(CAPTCHA_BOARD_LOG_DIR)
        os.mkdir(CAPTCHA_BOARD_LOG_DIR)

        merged = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter(CAPTCHA_BOARD_LOG_DIR, sess.graph)


        ckpt = tf.train.get_checkpoint_state(CAPTCHA_TRAIN_MOEDL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model : ", ckpt.model_checkpoint_path)

        try:
            step = 0
            while True:
                begin_time = datetime.now()
                batch_train_x_val, batch_train_y_val = sess.run([batch_train_x, batch_train_y])
                _, loss_, summary = sess.run([optimizer, loss, merged], feed_dict={X: batch_train_x_val, Y: batch_train_y_val, keep_prob: 0.75})
                end_time = datetime.now()
                print("captcha train step : %d loss : %f time : %fms" % (step, loss_, (end_time-begin_time).microseconds))

                step += 1
                if step % 1000 == 0:
                    batch_test_x_val, batch_test_y_val = sess.run([batch_test_x, batch_test_y])
                    accuracy_ = sess.run(accuracy, feed_dict={X: batch_test_x_val, Y: batch_test_y_val, keep_prob: 0.9})
                    print("captcha test %d accuracy : %f " % (step, accuracy_))
                    saver.save(sess, CAPTCHA_TRAIN_MOEDL_DIR+ "captcha.model", global_step=step)

                log_writer.add_summary(summary, step)
        except tf.errors.OutOfRangeError:
            print('done!')


if __name__ == '__main__':
    captcha_train_cnn()
