import tensorflow as tf
from tensorflow_cnn_train import *
from captcha_datasets import *

MODEL_DIR = './captcha_models/'

def captcha_eval(captcha_image):

    output = crack_captcha_cnn()

    predict = tf.reshape(output, [-1, CAPTCHA_MAX_CAPTCHA, CAPTCHA_CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_MAX_CAPTCHA, CAPTCHA_CHAR_SET_LEN]), 2)

    test_tfrecord_files = [CAPTCHA_TEST_TFRECORD + "test_captcha_" + str(i) + ".tfrecord" for i in
                           range(CAPTCHA_TEST_TFRECORD_COUNT)]
    batch_test_x, batch_test_y = get_dataset(test_tfrecord_files, 1, CAPTCHA_TEST_EPOCHS)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./captcha_models/"))

        try:

            while True:
                # image = ImageCaptcha()
                # captcha_text = random_captcha_text()
                # captcha_text = ''.join(captcha_text)
                # captcha_label = label2vector(captcha_text)
                # captcha_image = Image.open(image.generate(captcha_text))

                captcha_label = label2vector("7xrs")

                captcha_image = Image.open("C:/Users/86529/Desktop/7xrs.png")
                #captcha_image = Image.open("F:/demo/py/pydemo/captcha_demo/images/7xrs.png")
                captcha_image = np.array(captcha_image)
                captcha_image = rgb2gray(captcha_image)
                captcha_image = captcha_image.flatten() / 255

                batch_test_x_val = captcha_image.reshape([1, 9600])
                batch_test_y_val = captcha_label.reshape([1, 252])

                # batch_test_x_val, batch_test_y_val = sess.run([batch_test_x, batch_test_y])
                predict = tf.argmax(tf.reshape(output, [-1, 4, 63]), 2)
                max_idx_p_, max_idx_l_ = sess.run([max_idx_p, max_idx_l], feed_dict={X: batch_test_x_val, Y: batch_test_y_val, keep_prob: 1.0})
                # text = text_list[0].tolist()
                # vector = np.zeros(252)
                # i = 0
                # for n in text:
                #     vector[i * 4 + n] = 1
                #     i += 1
                #
                # nn = vector2label(vector)
                nnn = 0
        except tf.errors.OutOfRangeError:
            print('done!')

if __name__ == "__main__":

    # captcha_image = Image.open("F:/demo/py/pydemo/captcha_demo/images/qqNc.jpg")
    # captcha_image = captcha_image.resize((160, 60), Image.ANTIALIAS)
    # captcha_image = np.array(captcha_image)
    # captcha_image = rgb2gray(captcha_image)
    # captcha_image = captcha_image.flatten() / 255
    # captcha_image = captcha_image.reshape([1, 9600])
    captcha_image = 1
    captcha_eval(captcha_image)