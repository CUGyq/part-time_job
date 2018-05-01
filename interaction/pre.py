import tensorflow as tf
import numpy as np
import os
import train
import time
import cv2
MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
image_size = 128
img = cv2.imread("qqq.jpg")


img0 = cv2.resize(img, (128, 128))
img2 = tf.cast(img0,tf.float32)
img3 = tf.reshape(img2,(1,128,128,3))
y = train.inference(img3)
qq = tf.nn.softmax(y)

maxa = tf.argmax(y,1)
q = qq[0][maxa[0]]
variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
variable_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variable_to_restore)
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        for i in range(1):
            ss= sess.run(maxa)
            if ss == 0:
                cv2.rectangle(img, (20, 20), (img.shape[1] - 20, img.shape[0] - 20), (0, 255, 000), 2)
                cv2.putText(img, "normal！", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                            1)
                cv2.imshow("a", img)
                cv2.waitKey(0)
            elif ss == 1:
                cv2.rectangle(img, (20, 20), (img.shape[1] - 20, img.shape[0] - 20), (0, 0, 255), 2)
                cv2.putText(img, "abnormal", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                            1)
                cv2.imshow("a", img)
                cv2.waitKey(0)
    else:
        print("No checkpoint file found")

