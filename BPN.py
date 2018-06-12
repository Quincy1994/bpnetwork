# coding=utf-8

import tensorflow as tf
import numpy as np
from data_preprocess import *

def addLayer(inputData, inSize, outSize, activity_function = None):
    weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return weights, ans

# Load data
# ===========================================================
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# 在指定的间隔内返回均匀间隔的数字

X_train, y_train, X_dev, y_dev = read_dataset()


xs = tf.placeholder(tf.float32, [None, None])
ys = tf.placeholder(tf.float32, [None, 1])

weight1, layer1 = addLayer(xs, 10,5, activity_function=None)
weight2, layer2 = addLayer(layer1, 5, 1, activity_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-layer2)), reduction_indices=[1]))


train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train, feed_dict={xs:X_train, ys:y_train})
    sess.run(loss, feed_dict={xs:X_train, ys: y_train})
    if i% 50 == 0:
        print "Evalution:"
        print (sess.run(loss, feed_dict={xs:X_dev, ys:y_dev}))
        weight = sess.run(weight1, feed_dict={xs:X_dev, ys:y_dev})
        weight = np.concatenate(weight)
        print weight






























