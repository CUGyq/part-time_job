import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
SUMMARY_DIR = "F:/python/vedio/cnnlog/"
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 9
batch_size=50
# CONV1_DEEP = 10
CONV1_DEEP = 10
CONV1_SIZE = 3
# CONV2_DEEP = 16
CONV2_DEEP = 16
CONV2_SIZE = 3
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
num_train = 2001
MOVING_AVERAGE_DECAY = 0.99
FC_SIZE = 2048
MODEL_SAVE_PATH = "./cnnmodel/"
MODEL_NAME = "model.ckpt"
def variable_summaries(var,name ):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name,mean)
        stddev  = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name,stddev)
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
    with tf.name_scope('input'):
        tf.summary.image('input',img,batch_size)
    logit = inference(img)
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Moving_Average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope("loss_function"):
        # 交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # 正则L2表达式
        loss = cross_entropy_mean
        tf.summary.scalar('loss', loss)
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   3500 / batch_size, LEARNING_RATE_DECAY)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope("accuracy"):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logit, 1), label)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('acc', accuracy)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_train):
            summary,_,loss_Value, _, step = sess.run([merged,train_op, loss, train_step, global_step])
            writer.add_summary(summary,i)
            print("----------------------")
            print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
            if i%1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
                summary = sess.run(merged, options=run_options, run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                writer.add_summary(summary, i)
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            print("---------------------")
        coord.request_stop()
        coord.join(threads)
        writer.close()

def inference(input_img):
    with tf.name_scope('layer1-conv1'):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], stddev=0.1))
            variable_summaries(weights, 'layer1-conv1' + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[CONV1_DEEP]))
            variable_summaries(biases, "layer1-conv1" + '/biases')
        with tf.name_scope('plus_b'):
            conv = tf.nn.conv2d(input_img, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, biases ))
            tf.summary.histogram('layer1-conv1' + '/pre_activations',relu)
    with tf.name_scope('layer1-conv2'):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([CONV1_SIZE, CONV1_SIZE, CONV1_DEEP, CONV1_DEEP], stddev=0.1))
            variable_summaries(weights, 'layer1-conv2' + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[CONV1_DEEP]))
            variable_summaries(biases, "layer1-conv2" + '/biases')
        with tf.name_scope('plus_b'):
            conv = tf.nn.conv2d(relu, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, biases ))
            tf.summary.histogram('layer1-conv2' + '/pre_activations',relu)
    with tf.name_scope('layer2-pool1'):
        with tf.name_scope("pool1"):
            pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf.summary.histogram('layer2-pool1', pool)

    with tf.name_scope('layer3-conv2'):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], stddev=0.1))
            variable_summaries(weights, 'layer3-conv2' + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[CONV2_DEEP]))
            variable_summaries(biases, "layer1-conv1" + '/biases')
        with tf.name_scope('plus_b'):
            conv = tf.nn.conv2d(pool, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
            tf.summary.histogram('layer3-conv2' , relu)
            print(relu.get_shape())
    with tf.name_scope('layer3-conv3'):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([CONV2_SIZE, CONV2_SIZE, CONV2_DEEP, CONV2_DEEP], stddev=0.1))
            variable_summaries(weights, 'layer3-conv3' + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[CONV2_DEEP]))
            variable_summaries(biases, "layer3-conv3" + '/biases')
        with tf.name_scope('plus_b'):
            conv = tf.nn.conv2d(relu, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
            tf.summary.histogram('layer3-conv2' , relu)
    with tf.name_scope('layer4-pool2'):
        with tf.name_scope("pool2"):
            pool2 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf.summary.histogram('layer4-pool2', pool2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.name_scope('layer5-fc1'):
        with tf.name_scope("weight3"):
            weights3 = tf.Variable(tf.truncated_normal([nodes, FC_SIZE], stddev=0.1))
            variable_summaries(weights3, 'layer5-fc1' + '/weights')

        with tf.name_scope('biases3'):
            biases3 = tf.Variable(tf.constant(0.1, shape=[FC_SIZE]))
            variable_summaries(biases3, "layer5-fc1" + '/biases')
        with tf.name_scope('plus_b'):
            preactivate = tf.matmul(reshaped, weights3) + biases3
            tf.summary.histogram('layer5-fc1' + '/pre_activations', preactivate)
            fc1 = tf.nn.relu(preactivate, name='activation')
            tf.summary.histogram('layer5-fc1' + '/activations', fc1)
    with tf.name_scope('layer6-fc2'):
        with tf.name_scope("weight4"):
            weights4 = tf.Variable(tf.truncated_normal([FC_SIZE, NUM_LABELS], stddev=0.1))
            variable_summaries(weights4, 'layer6-fc2' + '/weights')
        with tf.name_scope('biases4'):
            biases4 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
            variable_summaries(biases4,"layer6-fc2"+'/biases')
        with tf.name_scope('plus_b'):
            out = tf.matmul(fc1,weights4) + biases4
            tf.summary.histogram('layer6-fc2' + '/pre_activations',out)
    return out
def main(argv=None):
    image_batch, label_batch = read_and_decode("train.tfrecords", batch_size=batch_size)

    train(image_batch,label_batch)


if __name__ == '__main__':
    tf.app.run()





