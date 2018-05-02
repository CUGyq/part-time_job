import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 5
batch_size=50
# CONV1_DEEP = 10
CONV1_DEEP = 10
CONV1_SIZE = 5
# CONV2_DEEP = 16
CONV2_DEEP = 16
CONV2_SIZE = 5
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
num_train = 5001
MOVING_AVERAGE_DECAY = 0.99
FC_SIZE = 2048
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32 / 255)
        image = tf.image.random_saturation(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32/255)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32 / 255)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)
def preprocess_for_train(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image,np.random.randint(2))
    return distorted_image
def read_and_decode(filename,batch_size):
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'channels': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })  # 取出包含image和label的feature对象
    image, label = features['img_raw'], features['label']
    height, width = features['height'], features['width']
    channels = features['channels']

    # height = tf.cast(height, tf.uint8)
    # width = tf.cast(width, tf.uint8)
    decoded_image = tf.decode_raw(image, tf.uint8)
    decoded_image = tf.reshape(decoded_image, [IMAGE_SIZE,IMAGE_SIZE,3])
    decoded_image = preprocess_for_train(decoded_image)
    min_after_dequeue = 100
    capacity = 1000 + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    image_batch = tf.cast(image_batch, tf.float32)
    return  image_batch,label_batch
def train(img,label):
    logit = inference(img)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # #正则L2表达式
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    loss = cross_entropy_mean
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 3500 / batch_size, LEARNING_RATE_DECAY)

    #train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # correct_prediction = tf.equal(tf.argmax(logit,1), label_batch)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:  # 开始一个会话
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_train):
            _,loss_Value, _, step = sess.run([train_op, loss, train_step, global_step])
            print("----------------------")
            print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
            if i%1000 == 0:
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            print("---------------------")

        coord.request_stop()
        coord.join(threads)
def inference(input_img):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_img, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        print(relu1.get_shape())
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(pool1.get_shape())
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        print(relu2.get_shape())
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(pool2.get_shape())
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    print(nodes)
    reshaped = tf.reshape(pool2, [-1, nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

def main(argv=None):
    image_batch, label_batch = read_and_decode("train.tfrecords", batch_size=batch_size)
    train(image_batch,label_batch)
    # test_batch, testlabel_batch = read_and_decode("train2.tfrecords", batch_size=10*batch_size)
    # evaluate(test_batch,testlabel_batch)

if __name__ == '__main__':
    tf.app.run()

