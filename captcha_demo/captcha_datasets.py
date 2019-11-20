from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import tensorflow as tf

CAPTCHA_TFRECORD_PATH = "./captcha_tfrecord/"
CAPTCHA_TFRECORD_COUNT = 10
CAPTCHA_IMAGE_COUNT = 10000
CAPTCHA_BATCH_SIZE = 32
CAPTCHA_THREADS = 4
CAPTCHA_CAPACITY = CAPTCHA_TFRECORD_COUNT*CAPTCHA_BATCH_SIZE*CAPTCHA_THREADS

# 每张图片(raw [n, RAW_LEN] [n, 9600], 宽高为60，160， 转换为灰度图之后降维为一维数组， 数组长度RAW_LEN（9600=60*160）)
# 每个标识（label [n, LABEL_LEN] [n, 256]， 4个字符,每个字符长度为CHAR_SET_LEN（63=10+26+26+1）,label总长度为LABEL_LEN(252=4*63)）
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
CAPTCHA_IMAGE_HEIGHT = 60
CAPTCHA_IMAGE_WIDTH = 160
CAPTCHA_CHAR_SET = number + alphabet + ALPHABET + ['_'];
CAPTCHA_CHAR_SET_LEN = len(CAPTCHA_CHAR_SET);
CAPTCHA_MAX_CAPTCHA = 4
CAPTCHA_RAW_LEN = CAPTCHA_IMAGE_HEIGHT*CAPTCHA_IMAGE_HEIGHT
CAPTCHA_LABEL_LEN = CAPTCHA_MAX_CAPTCHA*CAPTCHA_CHAR_SET_LEN


def rgb2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def label2vector(label):
    text_len = len(label)
    if text_len > CAPTCHA_MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(CAPTCHA_MAX_CAPTCHA*CAPTCHA_CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(label):
        idx = i * CAPTCHA_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vector2label(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CAPTCHA_CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))


def random_captcha_text(char_set = number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    image.write(captcha_text, "./images/" + captcha_text + '.jpg')  # 写到文件

    return captcha_text, image


def generate_tfrecord():
    for i in range(CAPTCHA_TFRECORD_COUNT):
        tfrecord_file = CAPTCHA_TFRECORD_PATH + "captcha_" + str(i) + ".tfrecord"
        with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
            for j in range(CAPTCHA_IMAGE_COUNT):
                image = ImageCaptcha()
                captcha_text = random_captcha_text()
                captcha_text = ''.join(captcha_text)

                captcha_label = label2vector(captcha_text)

                captcha_image = Image.open(image.generate(captcha_text))
                captcha_image = np.array(captcha_image)
                captcha_image = rgb2gray(captcha_image)
                captcha_image = captcha_image.flatten() / 255

                tf_example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'label': bytes_feature(captcha_label.tobytes()),
                        'raw': bytes_feature(captcha_image.tobytes())}))
                writer.write(tf_example.SerializeToString())
                print('generate tfrecord : %d %d' % (i, j))


def get_next_tfrecord():

    tfrecord_list = [CAPTCHA_TFRECORD_PATH + "captcha_" + str(i) + ".tfrecord" for i in range(10)]
    tfrecord_queue = tf.train.string_input_producer(tfrecord_list, shuffle=False)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'raw': tf.FixedLenFeature([], tf.string)})
    image_label = tf.decode_raw(features['label'], tf.float64)
    image_label = tf.reshape(image_label, [252,])

    image_raw = tf.decode_raw(features['raw'], tf.float64)
    image_raw = tf.reshape(image_raw, [9600, ])

    image_label, image_raw = tf.train.batch([image_label, image_raw],
                                            batch_size=CAPTCHA_BATCH_SIZE,
                                            capacity=CAPTCHA_CAPACITY,
                                            num_threads=CAPTCHA_THREADS)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        batch_x, batch_y = sess.run([image_label, image_raw])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    generate_tfrecord()

    # get_next_tfrecord()