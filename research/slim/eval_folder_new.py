from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import math
import tensorflow as tf
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import pdb

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'val', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def main(_):
    #print (open('testfile.txt','r').read()) 
    FILE = open('testfile.txt','a')
    with tf.Graph().as_default():
        dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        image_in = tf.placeholder(tf.uint8, shape=(299,299,3))

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        image = image_preprocessing_fn(image_in, eval_image_size, eval_image_size)
        image = tf.expand_dims(image, axis=0)
        logits, _ = network_fn(image)
        prediction = tf.argmax(logits, 1)

        variables_to_resotre = slim.get_variables_to_restore()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        with tf.Session() as sess:
            saver = tf.train.Saver(variables_to_resotre)
            saver.restore(sess, checkpoint_path)

            img_paths = glob.glob(FLAGS.dataset_dir + '/*.png')
            predictions = []
            avg_pool = tf.get_collection('avg_pool')[0]
            img_names, img_classes, img_percentage, img_features = [], [], [], []
            
            ###########################################
            #try loop to handle not having to check if file exists first.
            try:
              File = open('Output.txt','r')
              File.close()
            except:
              File = open('Output.txt','a')
              #write file headers
              for item in ['Filename','Prediction','actinedge','filopodia','hemisphere','lamellipodia','smallbleb']:
                File.write("%s " % item)
              File.write("\n")
              File.close()
            ###########################################            
            for file in img_paths:
                img = cv2.imread(file, 0)
                img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
                img = np.dstack((img,img,img))
                pre_logits, features = sess.run([logits, avg_pool], feed_dict={image_in:img})
                percentage = tf.nn.softmax(pre_logits).eval()[0]
                percentage=[round(percentage[x],8) for x in range(len(percentage))]
                prediction = tf.argmax(pre_logits, 1)
                prediction = prediction.eval()[0]
                img_name = file[file.rfind('/')+1:-4]
                print(img_name, prediction, percentage)
                ######################################
                File = open('Output.txt','a')
                File.write("%s " % img_name)
                File.write("%d " % prediction)
                for item in percentage:
                  File.write("%f " % item)
                File.write("\n")
                File.close()
                ######################################
                
                #img_names.append(img_name)
                #img_classes.append(prediction)
                #img_percentage.append(percentage)
                #img_features.append(features[0])
                #np.savez_compressed('./img_features.npz', img_names=img_names,
                #                    img_classes = img_classes,
                #                    img_percentage=img_percentage,
                #                    img_features=img_features)
                FILE.write('%s\n%s\n' % (img_name,percentage))
    FILE.close()


if __name__ == '__main__':
    tf.app.run()
