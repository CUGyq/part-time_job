import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image
import os
import random
import os
CHAR_SET_LEN = 10
CAPTCHA_LEN = 4
NUM_CHANNELS = 1
NUM_LABELS = CHAR_SET_LEN * CAPTCHA_LEN
batch_size=50

TRAINING_IMAGE_NAME = []


VALIDATION_IMAGE_NAME = []
#验证码图片的存放路径
CAPTCHA_IMAGE_PATH = './train/'
#验证码图片的宽度
CAPTCHA_IMAGE_WIDHT = 160
#验证码图片的高度
CAPTCHA_IMAGE_HEIGHT = 60

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

CONV3_DEEP = 64
CONV3_SIZE = 5

TRAIN_IMAGE_PERCENT = 0.6
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
num_train = 5001
MOVING_AVERAGE_DECAY = 0.99
FC_SIZE = 1024
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
def get_image_file_name(imgPath=CAPTCHA_IMAGE_PATH):
    fileName = []
    total = 0
    for filePath in os.listdir(imgPath):
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        total += 1
    return fileName, total
def name2label(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label
def get_data_and_label(fileName, filePath=CAPTCHA_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_label = name2label(fileName[0:CAPTCHA_LEN])
    return image_data, image_label
def get_next_batch(batchSize=32, trainOrTest='train', step=0):
    batch_data = np.zeros([batchSize, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])
    fileNameList = TRAINING_IMAGE_NAME
    if trainOrTest == 'validate':
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)

    indexStart = step * batchSize

    for i in range(batchSize):

        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data, img_label = get_data_and_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label
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
    decoded_image = tf.reshape(decoded_image, [64,64,3])
    decoded_image = preprocess_for_train(decoded_image)
    min_after_dequeue = 100
    capacity = 1000 + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    image_batch = tf.cast(image_batch, tf.float32)
    return  image_batch,label_batch
def train():
    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1], name='x-input')

    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    logit = inference(x_input,keep_prob)
    # print(logit.get_shape())

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # # #正则L2表达式
    loss = cross_entropy_mean
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 3500 / batch_size, LEARNING_RATE_DECAY)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    predict = tf.reshape(logit, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')
    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:  # 开始一个会话
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        steps = 0
        for epoch in range(6000):

            train_data, train_label = get_next_batch(50, 'train', steps)
            _, losss = sess.run([train_step, loss], feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            print("ssssssssssssssssssssssssssssssssssssssssssss")
            print(epoch)
            print(losss)
            print("ssssssssssssssssssssssssssssssssssssssssssss")
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(100, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.996:
                    saver.save(sess, MODEL_SAVE_PATH + "crack_captcha.model", global_step=steps)
                    break
            steps += 1


def inference(input_img,keep_prob):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_img, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        relu1 = tf.nn.dropout(relu1, keep_prob)
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        relu2 = tf.nn.dropout(relu2, keep_prob)
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer5-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        relu3 = tf.nn.dropout(relu3, keep_prob)
    with tf.variable_scope('layer6-pool4'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool3, [-1, nodes])
    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

def main(argv=None):
    image_filename_list, total = get_image_file_name(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    # 打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    # 分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[: trainImageNumber]
    # 和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber:]
    train(TRAINING_IMAGE_NAME, VALIDATION_IMAGE_NAME)
    # print('Training finished')

if __name__ == '__main__':
    image_filename_list, total = get_image_file_name(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    # 打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    # 分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[: trainImageNumber]
    # 和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber:]
    train()
    # tf.app.run()


