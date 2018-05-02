import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def get_files(filename):
    classes = []
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        classes.append(train_class)

    for index,name in enumerate(classes):
        for pic in os.listdir(filename+name):
            class_train.append(filename+name+'/'+pic)
            label_train.append(index)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()  #shuffle the samples
    np.random.shuffle(temp)  #after transpose,
    # images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    writer = tf.python_io.TFRecordWriter("test.tfrecords")
    for i in range(len(image_list)):
        img = cv2.imread(image_list[i])
        img = cv2.resize(img, (128, 128))
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(channels),
            'label': _int64_feature(int(label_list[i])),
            'img_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
if __name__ == '__main__':
    get_files("./test/")