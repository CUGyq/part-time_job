import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
SUMMARY_DIR = "F:/python/vedio/bplog/"
BATCH_SIZE = 100
TRAINING_STEPS = 5001
INPUT_NODE = 64*64*3
OUTPUT_NODE = 9
LAYER1_NODE = 2048
batch_size = 100

LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./bpmodel/"
MODEL_NAME="model.ckpt"
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

    decoded_image = tf.decode_raw(image, tf.uint8)
    decoded_image = tf.reshape(decoded_image, [64*64*3])

    min_after_dequeue = 100
    capacity = 1000 + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([decoded_image, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    image_batch = tf.cast(image_batch, tf.float32)
    return  image_batch,label_batch
def inference(input,regularizer):
    with tf.name_scope("layer1"):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
            variable_summaries(weights, 'layer1' + '/weights')
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(weights))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[LAYER1_NODE]))
            variable_summaries(biases,"layer1"+'/biases')
        with tf.name_scope('plus_b'):
            preactivate = tf.matmul(input,weights) + biases
            tf.summary.histogram('layer1' + '/pre_activations',preactivate)
            relu1 = tf.nn.relu(preactivate, name='activation')
            tf.summary.histogram('layer1' + '/activations', relu1)

    with tf.name_scope("layer2"):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
            variable_summaries(weights, 'layer2' + '/weights')
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(weights))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[OUTPUT_NODE]))
            variable_summaries(biases,"layer2"+'/biases')
        with tf.name_scope('plus_b'):
            out = tf.matmul(relu1,weights) + biases
            tf.summary.histogram('layer2' + '/pre_activations',out)
            return out

def variable_summaries(var,name ):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name,mean)
        stddev  = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name,stddev)

def train(img,label):
    with tf.name_scope('input'):
        image_shaped_input = tf.reshape(img, [-1, 64, 64, 3])
        tf.summary.image('input',image_shaped_input,batch_size)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(img,regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Moving_Average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope("loss_function"):
        # 交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # 正则L2表达式
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
        tf.summary.scalar('loss', loss)
    with tf.name_scope("train_step"):
        # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
        #                                           10000 / batch_size, LEARNING_RATE_DECAY)
        # tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1),label)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar('accuracy',accuracy)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            summary, _, loss_Value, _, step = sess.run([merged, train_op, loss, train_step, global_step])
            writer.add_summary(summary, i)
            print("----------------------")
            print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
            if i % 1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
                summary = sess.run(merged, options=run_options, run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                writer.add_summary(summary, i)
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            print("---------------------")
        coord.request_stop()
        coord.join(threads)
        writer.close()

def main(argv = None):
    image_batch, label_batch = read_and_decode("train.tfrecords", batch_size=batch_size)
    train(image_batch,label_batch)
if __name__ == '__main__':
    tf.app.run()

