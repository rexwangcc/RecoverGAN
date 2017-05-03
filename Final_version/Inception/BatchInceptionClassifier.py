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
import multiprocessing
import urllib
# import matplotlib.pyplot as plt
import glob
import sys
import tarfile
import re


class inceptionv4_classifier(object):

    def __init__(self,
                 extension='.jpg',
                 path_to_validate='to_validate/',
                 checkpoints_dir='checkpoints/',
                 keyword='cat',
                 top_k=5,
                 print_flag=False,
                 save_result_to_file=True,
                 pool_size=4,
                 delete_failed_images=True):

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

        # Initialize a (iamge, probability) tuple list for later storage
        self.image_and_probabilities = []

        # The keywords to judge
        if not isinstance(keyword, list):
            self.keyword = []
            self.keyword.append(keyword)
        else:
            self.keyword = keyword

        # The flag to determine whether print out each classification result
        self.print_flag = print_flag

        # How harsh the filter is, 1 is most harsh, 5 is least strict
        self.top_k = top_k

        # Initialize the accuracy over all
        self.accuracy = 0

        # The flag to determine whether to save results to a file
        self.save_result_to_file = save_result_to_file

        # Use the pool size of multiprocessing to create a Pool
        self.pool_size = pool_size

        # Initiate a list to store images that are classified as no-related to
        # keywords
        self.classcified_fail = []

        # The flag to determine whether remove images that are classified as
        # no-related to keywords by the end
        self.delete_failed_images = delete_failed_images

    def image_preprocessor(self, img):
        ''' A function to deal with a single image '''
        try:
            # Transform an iamge file to a encoded string using urllib
            # Note thses few steps may cost both time and space resources!! Should be
            # modified if possible
            img_str = urlopen(
                'file:' + urllib.request.pathname2url(img)).read()

            # Decode image string to a tensor
            image = tf.image.decode_jpeg(img_str, channels=3)

            # Use inception utils to preprocess image tensor
            processed_image = inception_preprocessing.preprocess_image(
                image, self.image_size, self.image_size, is_training=False)
            processed_images = tf.expand_dims(processed_image, 0)

            # return a tuple of (tensor, tensor, iamge_path)
            yield image, processed_images, img
        except:
            pass

    def remove_image_file(self, img):
        try:
            os.remove(img)
        except:
            print('Deletion for ' + img + ' failed!')

    def main(self):

        # Define a Tensorflow graph
        with tf.Graph().as_default():

            # Map the image_preprocessor over all files
            # p = multiprocessing.Pool(self.pool_size)

            self.processed_tensor_list = map(
                self.image_preprocessor, self.files)

            self.dim = len(list(self.processed_tensor_list))
            print('Total files to perform validation: ' + str(self.dim))
            print('--> This number may change if some files encoded improperly and throw exceptions during classifying <--')

            # Iterate over the map object
            for tensor_tuple in self.processed_tensor_list:
                print('*********************')
                # Create the model, use the default arg scope to configure the
                # batch norm parameters.
                with self.slim.arg_scope(inception4.inception_v4_arg_scope()):
                    logits, _ = inception4.inception_v4(
                        tensor_tuple[1], num_classes=1001, is_training=False)

                # Append a tuple (image, probability, image_path)
                self.image_and_probabilities.append(
                    (tensor_tuple[0], tf.nn.softmax(logits), tensor_tuple[2]))

            # Load the pretrained model
            self.init_fn = self.slim.assign_from_checkpoint_fn(
                os.path.join(self.checkpoints_dir, 'inception_v4.ckpt'),
                self.slim.get_model_variables('InceptionV4'))

            with tf.Session() as sess:
                self.init_fn(sess)
                for idx in range(self.dim):
                    print('Classifying on image' + str(idx))

                    try:
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

                        # Check if iamge is classified as the keywords
                        if any(keys in temp_array for keys in self.keyword):
                            self.accuracy += 1
                        else:
                            self.classcified_fail.append(
                                self.image_and_probabilities[idx][2])
                    except:
                        print('Exception occurred on image ' + str(idx))
                        self.dim -= 1
                        self.classcified_fail.append(
                            self.image_and_probabilities[idx][2])

        print('Classification Accuracy ====> ' +
              str(tf.divide(self.accuracy, self.dim)))

        if self.save_result_to_file:
            with open('Inception_v4_Results.txt', 'wb') as f:
                f.write('Files before inception\n'.encode())
                f.write(str(self.dim).encode())

                f.write('Files after inception\n'.encode())
                f.write(str(self.accuracy).encode())

                f.write('Classification Accuracy\n'.encode())
                f.write(str(tf.divide(self.accuracy, self.dim)).encode())

        if self.delete_failed_images:
            # p = multiprocessing.Pool(self.pool_size)
            map(self.remove_image_file, self.classcified_fail)


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
                 delete_failed_images=True,
                 save_result_to_file=True):

        self.label_lookup = label_lookup
        self.uid_lookup_path = uid_lookup_path
        self.model_path = model_path
        self.keyword = keyword
        self.filter_level = filter_level
        # self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.delete_failed_images = delete_failed_images
        # The flag to determine whether to save results to a file
        self.save_result_to_file = save_result_to_file
        # Initialize the accuracy over all
        self.accuracy = 0

        self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        if (path_to_pretrained_model is None) and (download_model == False):
            raise ValueError(
                'You didn\'t set a path to the pretrained_model, maybe you want to set "download_model" to True and let this script downloads it automatically?')
        if (path_to_pretrained_model is not None) and (download_model != False):
            raise ValueError('You set both path to the pretrained_model and "download_model" to True at the same time, maybe you want to set "download_model" to True and let this script downloads it automatically or just set the path?')
        if path_to_pretrained_model:
            if not path_to_pretrained_model.endswith('/'):
                self.Model_Save_Path = path_to_pretrained_model + '/'
            else:
                self.Model_Save_Path = path_to_pretrained_model
        self.Model_Save_Path = path_to_pretrained_model

        if download_model == True:
            self.maybe_download_and_extract(self.DATA_URL)

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

        self.dim = len(self.files)
        print('Total files to perform validation: ' + str(self.dim))

        self.tensor_files = map(self.convert_file_to_tensor, self.files)

    def maybe_download_and_extract(self, DATA_URL):
        """Download and extract model tar file."""

        print('Will download the pre-trained Inception Model to the same path with this validator!')
        self.Model_Save_Path = os.path.join("/",
                                            os.getcwd(), 'DownLoaded_Inception/')
        print('Start download to ' + self.Model_Save_Path)

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
        print('Inception v3 pre-trained model loaded successfully!')

    def run_inference_on_image(self):
        # Creates graph from saved GraphDef.
        self.load_inception()
        # To escape from the error:  'Could not allocate GPU device memory for device 0'/'CUDA_ERROR_OUT_OF_MEMORY'
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)

        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as
        # sess:
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            print('Making classifications:')

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup(label_lookup_path=self.Model_Save_Path + self.label_lookup,
                                     uid_lookup_path=self.Model_Save_Path + self.uid_lookup_path)
            current_counter = 1
            for (tensor_image, image) in self.tensor_files:
                print('On ' + str(current_counter))

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

                    print(this_prediction)

                    if not any(keyword in this_prediction for keyword in self.keyword):
                        if self.delete_failed_images:
                            os.remove(image)
                        else:
                            pass
                    else:
                        self.accuracy += 1

                except InvalidArgumentError:
                    print(
                        'Something wrong occuried during classification! Maybe the image is broken!')
                    print('Will delete it now!!!')
                    # time.sleep(0.8)
                    if self.delete_failed_images:
                        os.remove(image)
                    else:
                        pass
                    #     pass

                current_counter += 1

        if self.save_result_to_file:
            with open('Inception_v3_Results.txt', 'wb') as f:
                f.write('Files before inception\n'.encode())
                f.write(str(self.dim).encode())

                f.write('Files after inception\n'.encode())
                f.write(str(self.accuracy).encode())

                f.write('Classification Accuracy\n'.encode())
                f.write(str(tf.divide(self.accuracy, self.dim)).encode())

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
    inceptionv3_task = Advanced_validate(path_to_validate='/',
                                         path_to_pretrained_model=None,
                                         download_model=True,
                                         label_lookup="imagenet_2012_challenge_label_map_proto.pbtxt",
                                         uid_lookup_path="imagenet_synset_to_human_label_map.txt",
                                         model_path="classify_image_graph_def.pb",
                                         extension='jpg',
                                         keyword=['plane', 'aircraft'],
                                         filter_level=1,
                                         delete_failed_images=True,
                                         save_result_to_file=True)
    inceptionv3_task.run()

    # inceptionv4_task = inceptionv4_classifier(extension='.jpg',
    #                                           path_to_validate='/',
    #                                           checkpoints_dir='checkpoints/',
    #                                           keyword=['plane', 'aircraft'],
    #                                           top_k=1,
    #                                           print_flag=False,
    #                                           save_result_to_file=True,
    #                                           pool_size=8,
    #                                           delete_failed_images=True
    #                                           )
    # inceptionv4_task.main()
