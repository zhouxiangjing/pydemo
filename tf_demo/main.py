
import tensorflow as tf

print(tf.__version__)

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.3, 0.4]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

sess.run(w1.initializer)
sess.run(w2.initializer)

print(sess.run(w1))
print(sess.run(w2))
print(sess.run(a))
print(sess.run(y))
sess.close()