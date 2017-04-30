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

        # Import Tensorflow Slim
        self.slim = tf.contrib.slim

        # The download URL of pretrained Incption model
        self.model_url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"

        # Check if model directory exists, if not create one
        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir

        # Check if model file exists, if not download it to 'checkpoints_dir'
        if not tf.gfile.Exists(checkpoints_dir + 'inception_v4_2016_09_09.tar.gz'):
            dataset_utils.download_and_uncompress_tarball(
                self.model_url, self.checkpoints_dir)

        # Set Image size to Inception default size
        self.image_size = inception4.inception_v4.default_image_size

        # Set the image file extension, only support 1 kind of image for 1 time
        self.extension = extension

        # Check if the directory contains images to be validated exists, if not
        # create one
        if not tf.gfile.Exists(path_to_validate):
            tf.gfile.MakeDirs(path_to_validate)
        self.path_to_validate = path_to_validate

        # Load all paths of images to be validated to a list
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
        if not tf.gfile.Exists(path_to_validate):
            tf.gfile.MakeDirs(path_to_validate)
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
        if not tf.gfile.Exists(path_to_validate):
            tf.gfile.MakeDirs(path_to_validate)
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


class Advanced_validate(object):

    def __init__(self,
                 path_to_validate,
                 path_to_pretrained_model=None,
                 download_model=False,
                 label_lookup="imagenet_2012_challenge_label_map_proto.pbtxt",
                 uid_lookup_path="imagenet_synset_to_human_label_map.txt",
                 model_path="classify_image_graph_def.pb",
                 extension=None,
                 keyword=['cat'],
                 filter_level=5,
                 per_process_gpu_memory_fraction=0.5):

        self.label_lookup = label_lookup
        self.uid_lookup_path = uid_lookup_path
        self.model_path = model_path
        self.keyword = keyword
        self.filter_level = filter_level
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        if (path_to_pretrained_model is None) and (download_model == False):
            raise ValueError(
                'You didn\'t set a path to the pretrained_model, maybe you want to set "download_model" to True and let this script downloads it automatically?')
        if (path_to_pretrained_model is not None) and (download_model != False):
            raise ValueError('You set both path to the pretrained_model and "download_model" to True at the same time, maybe you want to set "download_model" to True and let this script downloads it automatically or just set the path?')

        if not path_to_pretrained_model.endswith('/'):
            self.Model_Save_Path = path_to_pretrained_model + '/'
        else:
            self.Model_Save_Path = path_to_pretrained_model
        self.Model_Save_Path = path_to_pretrained_model

        if download_model == True:
            self.maybe_download_and_extract()

        if not os.path.exists(path_to_validate):
            raise ValueError(
                "Confirm you specified the right path including the images!")

        if not path_to_validate.endswith('/'):
            self.path_to_validate = path_to_validate + '/'
        else:
            self.path_to_validate = path_to_validate

        if extension is None:
            self.extension = ''
        elif not extension.startswith('.'):
            self.extension = '.' + extension
        else:
            self.extension = extension

        self.files = [filename for filename in glob.glob(
            self.path_to_validate + '*' + self.extension)]

        print 'Total files to perform validation: ' + str(len(self.files))

        self.tensor_files = map(self.convert_file_to_tensor, self.files)

    def maybe_download_and_extract(self):
        """Download and extract model tar file."""

        print 'Will download the pre-trained Inception Model to the same path with this validator!'
        self.Model_Save_Path = os.path.join(
            os.getcwd(), './DownLoaded_Inception/')
        print 'Start download to ' + self.Model_Save_Path

        if not os.path.exists(self.Model_Save_Path):
            os.makedirs(self.Model_Save_Path)

        dest_directory = self.Model_Save_Path

        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(
                DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename,
                  statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def convert_file_to_tensor(self, src_file):
        # Return a (tensor,src_file) tuple
        return (tf.gfile.FastGFile(src_file, 'rb').read(), src_file)

    def load_inception(self):
        with tf.gfile.FastGFile(self.Model_Save_Path + self.model_path) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        print 'Inception v3 pre-trained model loaded successfully!'

    def run_inference_on_image(self):
        # Creates graph from saved GraphDef.
        self.load_inception()
        # To escape from the error:  'Could not allocate GPU device memory for device 0'/'CUDA_ERROR_OUT_OF_MEMORY'
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as
        # sess:
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            print 'Making classifications:'

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup(label_lookup_path=self.Model_Save_Path + self.label_lookup,
                                     uid_lookup_path=self.Model_Save_Path + self.uid_lookup_path)
            current_counter = 1
            for (tensor_image, image) in self.tensor_files:
                print 'On ' + str(current_counter)

                try:
                    predictions = sess.run(
                        softmax_tensor, {'DecodeJpeg/contents:0': tensor_image})
                    predictions = np.squeeze(predictions)

                    top_k = predictions.argsort(
                    )[-int(self.filter_level):][::-1]

                    # for node_id in top_k:
                    #     human_string = node_lookup.id_to_string(node_id)
                    #     score = predictions[node_id]

                    # this_prediction = (node_lookup.id_to_string(top_k[0]), predictions[top_k[0]])

                    this_prediction = node_lookup.id_to_string(top_k[0])

                    print this_prediction

                    # if self.keyword not in this_prediction:
                    # os.remove(image)

                    if not any(keyword in this_prediction for keyword in self.keyword):
                        os.remove(image)
                    #
                    # Dirty code for now!!!!!!!!!!!!
                    # else:
                    #     shutil.copy(image, '/home/rex/cs585/data/perfect_dogs2/')

                    # time.sleep(0.8)
                    # To catch the error:
                    # tensorflow.python.framework.errors.InvalidArgumentError:
                    # Invalid JPEG data, size 2051
                except InvalidArgumentError:
                    print 'Something wrong occuried during classification! Maybe the image is broken!'
                    print 'Will delete it now!!!'
                    # time.sleep(0.8)
                    os.remove(image)
                    #     pass

                current_counter += 1

#-----
        # # To escape from the error:  'Could not allocate GPU device memory for device 0'/'CUDA_ERROR_OUT_OF_MEMORY'
        # # config=tf.ConfigProto(gpu_options=gpu_options)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # sess.run()
        # try:
        #     with tf.Session() as sess:
        #         softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        #         print 'Making classifications:'

        #         current_counter = 1
        #         for (tensor_image, image) in self.tensor_files:
        #             print 'On ' + str(current_counter)

        #             try:
        #                 predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': tensor_image})
        #                 predictions = np.squeeze(predictions)

        #                 # Creates node ID --> English string lookup.
        #                 node_lookup = NodeLookup(label_lookup_path=self.Model_Save_Path + self.label_lookup,
        # uid_lookup_path=self.Model_Save_Path + self.uid_lookup_path)

        #                 top_k = predictions.argsort()[-int(self.filter_level):][::-1]

        #                 # for node_id in top_k:
        #                 #     human_string = node_lookup.id_to_string(node_id)
        #                 #     score = predictions[node_id]

        #                 # this_prediction = (node_lookup.id_to_string(top_k[0]), predictions[top_k[0]])

        #                 this_prediction = node_lookup.id_to_string(top_k[0])

        #                 print this_prediction

        #                 # if self.keyword not in this_prediction:
        #                 #     os.remove(image)
        #                 if not any(keyword in this_prediction for keyword in self.keyword):
        #                     os.remove(image)
        #             except:
        #                 print 'Something wrong occuried during classification! Maybe the image is broken!'
        #                 pass

        #             current_counter += 1
        # finally:
        #     sess.close()
#-----

    def run(self):
        self.run_inference_on_image()


class NodeLookup(object):

    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):

        if (not label_lookup_path) or (not os.path.exists(label_lookup_path)):
            raise ValueError(
                'Confirm you del node_lookupspecified the correct full-path to the inception mapping file: "imagenet_2012_challenge_label_map_proto.pbtxt"!')
        if not uid_lookup_path:
            raise ValueError(
                'Confirm you specified the correct full-path to the inception mapping file: "imagenet_synset_to_human_label_map.txt"!')

        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


if __name__ == '__main__':
    pass
