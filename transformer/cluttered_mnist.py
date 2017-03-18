# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from datetime import datetime
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from random import randint
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from tqdm import *

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('number_of_loc_nets', 10,
                            """Number of localisation nets.""")
tf.app.flags.DEFINE_float('sigma', 0.1,
                            """Number of localisation nets.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_boolean('use_fp16', False,
#                             """Train the model using fp16.""")


# Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

#  Graph representation of our network

#  Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])

#  We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
mask_placeholder = tf.placeholder(tf.bool, [FLAGS.number_of_loc_nets])


def localisation_net(x, keep_prob):
    with tf.variable_scope('localisation_net'):
        # Since x is currently [batch, height*width], we need to reshape to a
        # 4-D tensor to use it in a convolutional graph.  If one component of
        # `shape` is the special value -1, the size of that dimension is
        # computed so that the total size remains constant.  Since we haven't
        # defined the batch dimension's shape yet, we use -1 to denote this
        # dimension should not change size.
        x_tensor = tf.reshape(x, [-1, 40, 40, 1])

        #  We'll setup the two-layer localisation network to figure out the
        #  parameters for an affine transformation of the input
        #  Create variables for fully connected layer
        W_fc_loc1 = weight_variable([1600, 20])
        b_fc_loc1 = bias_variable([20])

        W_fc_loc2 = weight_variable([20, 6])
        # Use identity transformation as starting point
        noise = np.random.normal(loc=0.0, scale=FLAGS.sigma, size=[2, 3])
        initial = np.array([[1., 0, 0], [0, 1., 0]]) + noise
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

        #  Define the two layer localisation network
        h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
        h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
        #  Second layer
        h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

        #  We'll create a spatial transformer module to identify discriminative
        #  patches
        out_size = (40, 40)
        h_trans = transformer(x_tensor, h_fc_loc2, out_size)
        return h_trans, h_fc_loc2


def create_net(h_trans, name):
    with tf.variable_scope(name):
        #  We'll setup the first convolutional layer
        # Weight matrix is [height x width x input_channels x output_channels]
        filter_size = 3
        n_filters_1 = 16
        W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

        #  Bias is [output_channels]
        b_conv1 = bias_variable([n_filters_1])

        #  Now we can build a graph which does the first layer of convolution:
        # we define our stride as batch x height x width x channels
        # instead of pooling, we use strides of 2 and more layers
        # with smaller filters.

        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(input=h_trans,
                        filter=W_conv1,
                        strides=[1, 2, 2, 1],
                        padding='SAME') +
            b_conv1)

        #  And just like the first layer, add additional layers to create
        # a deep net
        n_filters_2 = 16
        W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
        b_conv2 = bias_variable([n_filters_2])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(input=h_conv1,
                        filter=W_conv2,
                        strides=[1, 2, 2, 1],
                        padding='SAME') +
            b_conv2)

        #  We'll now reshape so we can connect to a fully-connected layer:
        h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])

        #  Create a fully-connected layer:
        n_fc = 1024
        W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
        b_fc1 = bias_variable([n_fc])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #  And finally our softmax layer:
        W_fc2 = weight_variable([n_fc, 10])
        b_fc2 = bias_variable([10])
        y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_logits

h_trans = []
h_fc_loc2 = []
for idx in range(FLAGS.number_of_loc_nets):
    h_trans_temp, h_fc_loc2_temp = localisation_net(x, keep_prob)
    h_trans.append(h_trans_temp)
    tf.summary.image('h_trans_{}'.format(idx), h_trans_temp)
    h_fc_loc2.append(h_fc_loc2_temp)

# with tf.variable_scope('net') as scope:
#     y_logits_list = []
#     for idx in range(FLAGS.number_of_loc_nets):
#         if idx > 0:
#             scope.reuse_variables()
#         y_logits_temp = create_net(h_trans[idx], 'net_{}'.format(idx))
#         y_logits_list.append(y_logits_temp)
#     y_logits = tf.reduce_mean(tf.stack(y_logits_list), axis=0)
#     print(y_logits.get_shape())

with tf.variable_scope('net') as scope:
    print(h_trans[0].get_shape())
    print(tf.stack(h_trans)[0].get_shape())
    h_trans_masked = tf.squeeze(
        tf.boolean_mask(tf.stack(h_trans), mask_placeholder),
        axis=0)
    print(h_trans_masked.get_shape())
    y_logits = create_net(h_trans_masked, 'net_{}'.format(idx))

#  Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
# grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

#  Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

# Monitor h divergence
mean_h_fc_loc2 = tf.reduce_mean(tf.stack(h_fc_loc2), axis=0)
print('mean_h_fc_loc2 shape: ',mean_h_fc_loc2.get_shape())
div = tf.reduce_sum(tf.square(tf.subtract(h_fc_loc2, mean_h_fc_loc2)))
tf.summary.scalar('div', div)

merged = tf.summary.merge_all()


#  We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
train_writer = tf.summary.FileWriter(
    './train/locnets={}/sigma={}'.format(
        FLAGS.number_of_loc_nets,
        FLAGS.sigma),
    sess.graph)
#  We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = 100
n_epochs = 500
train_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in tqdm(range(iter_per_epoch - 1)):
        batch_xs = X_train[indices[iter_i]:indices[iter_i + 1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i + 1]]
        mask = [False] * FLAGS.number_of_loc_nets
        mask[randint(0, FLAGS.number_of_loc_nets - 1)] = True
        mask = np.array(mask)
        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                mask_placeholder: mask,
                                keep_prob: 1.0})
            print('Epoch: ' + str(epoch_i) + ', Iteration: ' +
                  str(iter_i) + ', Loss: ' + str(loss))

        sess.run(optimizer,
                 feed_dict={
                     x: batch_xs,
                     y: batch_ys,
                     keep_prob: 0.8,
                     mask_placeholder: mask})

    summary, acc = sess.run([merged, accuracy],
                            feed_dict={
                                x: X_valid,
                                y: Y_valid,
                                mask_placeholder: mask,
                                keep_prob: 1.0})
    print('Accuracy (%d): ' % epoch_i + str(acc))
    train_writer.add_summary(summary, epoch_i)
    theta = sess.run(h_fc_loc2,
                     feed_dict={x: batch_xs, keep_prob: 1.0})
    for idx in range(FLAGS.number_of_loc_nets):
        print(theta[idx][0])
