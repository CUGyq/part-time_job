import cv2
import numpy as np
import os.path
import tensorflow as tf
import train_cnn
import time


MOVING_AVERAGE_DECAY = 0.99
EVAL_INTERVAL_SECS = 10
c = 0
cap = cv2.VideoCapture('data/bend/daria_bend.avi')

while (cap.isOpened()):
    c += 1
    ret, frame = cap.read()
    if (ret == 0):
        break;
    img0 = cv2.resize(frame, (128, 128))
    with tf.Graph().as_default() as g:
        img2 = tf.cast(img0, tf.float32)
        img3 = tf.reshape(img2, (1, 128, 128, 3))
        logit = train_cnn.inference(img3)
        qq = tf.nn.softmax(logit)
        maxa = tf.argmax(logit, 1)
        q = qq[0][maxa[0]]
        variable_averages = tf.train.ExponentialMovingAverage(train_cnn.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(train_cnn.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                ss, dd = sess.run([maxa, q])
                cv2.rectangle(frame, (20, 20), (frame.shape[1]-20, frame.shape[0] - 20), (0, 255, 0), 2)

                if ss == 0:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "bend", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),
                                1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 1:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "jack", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,0),
                                    1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)

                elif ss == 2:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "jump" , (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                                    1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 3:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "pjump", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 4:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "run", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                                1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 5:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "side", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                                1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 6:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "skip", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                elif ss == 7:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "walk", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                                1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)
                else:
                    cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (255, 0, 0), 2)
                    cv2.putText(frame, "wave1", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                                1)
                    cv2.imshow("a", frame)
                    cv2.waitKey(1)

            else:
                print("No checkpoint file found")



