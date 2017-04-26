# Batchable Classifier based on Inception and Tensorflow Slim
# @Rex Wang @Yt Su

from models import dataset_utils
from models import imagenet
from models import inception_preprocessing
from models import inception_v4 as inception4
from models import inception_v3 as inception3
from models import inception_v2 as inception2

import numpy as np
import os
import tensorflow as tf
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt
import glob


class inceptionv4_classifier(object):

    def __init__(self, extension='.jpg', path_to_validate='to_validate/', checkpoints_dir='checkpoints/', keyword='cat', top_k=5, print_flag=False, save_result_to_file=True):
        print('***Running Claasifier with [Inception-v4] core***')
        self.slim = tf.contrib.slim
        self.model_url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir
        if not tf.gfile.Exists(checkpoints_dir + 'inception_v4_2016_09_09.tar.gz'):
            dataset_utils.download_and_uncompress_tarball(
                self.model_url, self.checkpoints_dir)

        self.image_size = inception4.inception_v4.default_image_size
        self.extension = extension
        self.path_to_validate = path_to_validate
        self.files = [filename for filename in glob.glob(
            self.path_to_validate + '*' + self.extension)]
        self.dim = len(self.files)
        print('Total files to perform validation: ' + str(self.dim))

        self.image_and_probabilities = []
        self.keyword = keyword
        self.print_flag = print_flag
        self.top_k = top_k
        self.accuracy = 0
        self.save_result_to_file = save_result_to_file

    def image_preprocessor(self, img):
        img_str = urlopen('file:' + urllib.request.pathname2url(img)).read()
        image = tf.image.decode_jpeg(img_str, channels=3)
        processed_image = inception_preprocessing.preprocess_image(
            image, self.image_size, self.image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
        # return a tuple of (tensor, tensor)
        return image, processed_images

    def main(self):

        with tf.Graph().as_default():
            self.processed_tensor_list = map(
                self.image_preprocessor, self.files)

            # Iterate over a map object
            for tensor_tuple in self.processed_tensor_list:

                # Create the model, use the default arg scope to configure the
                # batch norm parameters.
                with self.slim.arg_scope(inception4.inception_v4_arg_scope()):
                    logits, _ = inception4.inception_v4(
                        tensor_tuple[1], num_classes=1001, is_training=False)
                # Append a tuple (image, probability)
                self.image_and_probabilities.append(
                    (tensor_tuple[0], tf.nn.softmax(logits)))

            self.init_fn = self.slim.assign_from_checkpoint_fn(
                os.path.join(self.checkpoints_dir, 'inception_v4.ckpt'),
                self.slim.get_model_variables('InceptionV4'))

            with tf.Session() as sess:
                self.init_fn(sess)
                for idx in range(self.dim):
                    print('Classifying on image' + str(idx))
                    _, probabilities = sess.run([self.image_and_probabilities[idx][
                                                0], self.image_and_probabilities[idx][1]])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(
                        enumerate(-probabilities), key=lambda x:x[1])]

                    names = imagenet.create_readable_names_for_imagenet_labels()

                    temp_array = []
                    for i in range(self.top_k):
                        index = sorted_inds[i]
                        temp_array.append(names[index])
                        if self.print_flag:
                            print('Probability %0.2f%% => [%s]' % (
                                probabilities[index], names[index]))
                    if any(self.keyword in s for s in temp_array):
                        self.accuracy += 1
        print('Classification Accuracy ====> ' +
              str(tf.divide(self.accuracy, self.dim)))

        if self.save_result_to_file:
            with open('Inception_v4_Results.txt', 'wb') as f:
                f.write('Classification Accuracy\n')
                f.write(str(tf.divide(self.accuracy, self.dim)))


class inceptionv3_classifier(object):

    def __init__(self, extension='.jpg', path_to_validate='to_validate/', checkpoints_dir='checkpoints/', keyword='cat', top_k=5, print_flag=False, save_result_to_file=True):
        print('***Running Claasifier with [Inception-v3] core***')
        self.slim = tf.contrib.slim
        self.model_url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir
        if not tf.gfile.Exists(checkpoints_dir + 'inception_v3_2016_08_28.tar.gz'):
            dataset_utils.download_and_uncompress_tarball(
                self.model_url, self.checkpoints_dir)

        self.image_size = inception3.inception_v3.default_image_size
        self.extension = extension
        self.path_to_validate = path_to_validate
        self.files = [filename for filename in glob.glob(
            self.path_to_validate + '*' + self.extension)]
        self.dim = len(self.files)
        print('Total files to perform validation: ' + str(self.dim))

        self.image_and_probabilities = []
        self.keyword = keyword
        self.print_flag = print_flag
        self.top_k = top_k
        self.accuracy = 0
        self.save_result_to_file = save_result_to_file

    def image_preprocessor(self, img):
        img_str = urlopen('file:' + urllib.request.pathname2url(img)).read()
        image = tf.image.decode_jpeg(img_str, channels=3)
        processed_image = inception_preprocessing.preprocess_image(
            image, self.image_size, self.image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
        # return a tuple of (tensor, tensor)
        return image, processed_images

    def main(self):

        with tf.Graph().as_default():
            self.processed_tensor_list = map(
                self.image_preprocessor, self.files)

            # Iterate over a map object
            for tensor_tuple in self.processed_tensor_list:

                # Create the model, use the default arg scope to configure the
                # batch norm parameters.
                with self.slim.arg_scope(inception3.inception_v3_arg_scope()):
                    logits, _ = inception3.inception_v3(
                        tensor_tuple[1], num_classes=1001, is_training=False)
                # Append a tuple (image, probability)
                self.image_and_probabilities.append(
                    (tensor_tuple[0], tf.nn.softmax(logits)))

            self.init_fn = self.slim.assign_from_checkpoint_fn(
                os.path.join(self.checkpoints_dir, 'inception_v3.ckpt'),
                self.slim.get_model_variables('InceptionV3'))

            with tf.Session() as sess:
                self.init_fn(sess)
                for idx in range(self.dim):
                    print('Classifying on image' + str(idx))
                    _, probabilities = sess.run([self.image_and_probabilities[idx][
                                                0], self.image_and_probabilities[idx][1]])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(
                        enumerate(-probabilities), key=lambda x:x[1])]

                    names = imagenet.create_readable_names_for_imagenet_labels()

                    temp_array = []
                    for i in range(self.top_k):
                        index = sorted_inds[i]
                        temp_array.append(names[index])
                        if self.print_flag:
                            print('Probability %0.2f%% => [%s]' % (
                                probabilities[index], names[index]))
                    if any(self.keyword in s for s in temp_array):
                        self.accuracy += 1
        print('Classification Accuracy ====> ' +
              str(tf.divide(self.accuracy, self.dim)))

        if self.save_result_to_file:
            with open('Inception_v3_Results.txt', 'wb') as f:
                f.write('Classification Accuracy\n')
                f.write(str(tf.divide(self.accuracy, self.dim)))


class inceptionv2_classifier(object):

    def __init__(self, extension='.jpg', path_to_validate='to_validate/', checkpoints_dir='checkpoints/', keyword='cat', top_k=5, print_flag=False, save_result_to_file=True):
        print('***Running Claasifier with [Inception-v2] core***')
        self.slim = tf.contrib.slim
        self.model_url = "http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz"
        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir
        if not tf.gfile.Exists(checkpoints_dir + 'inception_v2_2016_08_28.tar.gz'):
            dataset_utils.download_and_uncompress_tarball(
                self.model_url, self.checkpoints_dir)

        self.image_size = inception2.inception_v2.default_image_size
        self.extension = extension
        self.path_to_validate = path_to_validate
        self.files = [filename for filename in glob.glob(
            self.path_to_validate + '*' + self.extension)]
        self.dim = len(self.files)
        print('Total files to perform validation: ' + str(self.dim))

        self.image_and_probabilities = []
        self.keyword = keyword
        self.print_flag = print_flag
        self.top_k = top_k
        self.accuracy = 0
        self.save_result_to_file = save_result_to_file

    def image_preprocessor(self, img):
        img_str = urlopen('file:' + urllib.request.pathname2url(img)).read()
        image = tf.image.decode_jpeg(img_str, channels=3)
        processed_image = inception_preprocessing.preprocess_image(
            image, self.image_size, self.image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
        # return a tuple of (tensor, tensor)
        return image, processed_images

    def main(self):

        with tf.Graph().as_default():
            self.processed_tensor_list = map(
                self.image_preprocessor, self.files)

            # Iterate over a map object
            for tensor_tuple in self.processed_tensor_list:

                # Create the model, use the default arg scope to configure the
                # batch norm parameters.
                with self.slim.arg_scope(inception2.inception_v2_arg_scope()):
                    logits, _ = inception2.inception_v4(
                        tensor_tuple[1], num_classes=1001, is_training=False)
                # Append a tuple (image, probability)
                self.image_and_probabilities.append(
                    (tensor_tuple[0], tf.nn.softmax(logits)))

            self.init_fn = self.slim.assign_from_checkpoint_fn(
                os.path.join(self.checkpoints_dir, 'inception_v2.ckpt'),
                self.slim.get_model_variables('InceptionV2'))

            with tf.Session() as sess:
                self.init_fn(sess)
                for idx in range(self.dim):
                    print('Classifying on image' + str(idx))
                    _, probabilities = sess.run([self.image_and_probabilities[idx][
                                                0], self.image_and_probabilities[idx][1]])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(
                        enumerate(-probabilities), key=lambda x:x[1])]

                    names = imagenet.create_readable_names_for_imagenet_labels()

                    temp_array = []
                    for i in range(self.top_k):
                        index = sorted_inds[i]
                        temp_array.append(names[index])
                        if self.print_flag:
                            print('Probability %0.2f%% => [%s]' % (
                                probabilities[index], names[index]))
                    if any(self.keyword in s for s in temp_array):
                        self.accuracy += 1
        print('Classification Accuracy ====> ' +
              str(tf.divide(self.accuracy, self.dim)))

        if self.save_result_to_file:
            with open('Inception_v2_Results.txt', 'wb') as f:
                f.write('Classification Accuracy\n')
                f.write(str(tf.divide(self.accuracy, self.dim)))
