import tensorflow as tf

TFRECORD_PATH = "./tfrecord/"

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/label': bytes_feature(""),
            'image/raw': bytes_feature("")}))

writer = tf.python_io.TFRecordWriter(TFRECORD_PATH)
writer.write(tf_example.SerializeToString())
writer.close()