# App module
import os
import numpy as np
import scipy.misc
import tensorflow as tf
import pprint


# import DCGAN class
from model import DCGAN
# import utils
from utils import show_all_variables


# Define a tensorflow app and the flags
flags = tf.app.flags
flags.DEFINE_integer(flag_name="epoch", default_value=20,
                     docstring="Epochs to train [20]")
flags.DEFINE_float(flag_name="learning_rate", default_value=0.0002,
                   docstring="Learning Rate for Gradient Optimizer [0.0002]")
flags.DEFINE_float(flag_name="beta1", default_value=0.5,
                   docstring="Momentum term of AdamOptimizer [0.5]")

flags.DEFINE_integer(flag_name="train_size", default_value=np.inf,
                     docstring="The size of train images [np.inf]")
flags.DEFINE_integer(flag_name="batch_size", default_value=64,
                     docstring="The size of batch images [64]")
flags.DEFINE_integer(flag_name="input_height", default_value=64,
                     docstring="The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer(flag_name="input_width", default_value=None,
                     docstring="The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer(flag_name="output_height", default_value=64,
                     docstring="The size of the output images to produce [64]")
flags.DEFINE_integer(flag_name="output_width", default_value=None,
                     docstring="The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer(flag_name="c_dim", default_value=3,
                     docstring="Dimension of image color/channels. [3]")
# DCGAN
# flags.DEFINE_string(flag_name="dataset", default_value="celebA", docstring = "The name of dataset [celebA, mnist, lsun]")

# Xz-GAN
##############################################################################################
flags.DEFINE_string(flag_name="dataset", default_value="celebA",
                    docstring="The name of dataset feed to Discriminator [celebA, mnist, lsun]")
flags.DEFINE_string(flag_name="dataset2", default_value="celebA",
                    docstring="The name of dataset feed to Generator [celebA, mnist, lsun]")
##############################################################################################


flags.DEFINE_string(flag_name="input_file_extension", default_value="*.jpg",
                    docstring="Glob pattern of filename of input images [*]")
flags.DEFINE_string(flag_name="checkpoint_dir", default_value="checkpoint",
                    docstring="Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string(flag_name="sample_dir", default_value="samples",
                    docstring="Directory name to save the image samples [samples]")

flags.DEFINE_boolean(flag_name="is_train", default_value=False,
                     docstring="True for training, False for testing [False]")
flags.DEFINE_boolean(flag_name="is_crop", default_value=False,
                     docstring="True for training, False for testing [False]")

flags.DEFINE_float(flag_name="gpu_utilization", default_value=0.8,
                   docstring="Per process GPU memory fraction [0.8]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    # Print out the parameters
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    # Deal with input/output size default values
    if not FLAGS.input_width:
        FLAGS.input_width = FLAGS.input_height
    if not FLAGS.output_width:
        FLAGS.output_width = FLAGS.output_height

    # Deal with checkpoint/sample directory path
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # Deal with GPU utilization
    def get_default_gpu_session(fraction=0.8):
        if fraction > 1:
            fraction = 0.8
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        return tf.Session(config=config)

    with get_default_gpu_session(FLAGS.gpu_utilization) as sess:
        # Deal with MNIST dataset
        if FLAGS.dataset == 'mnist':
            # Instantiate a dcgan isntance
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                c_dim=1,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_file_extension,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir
            )
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_file_extension,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                # Xx-GAN
                dataset_name2=FLAGS.dataset2
            )

        show_all_variables()

        # Deal with training
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception(
                    "[!!!] Need to train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
    '''
    python main.py --dataset test_cat_O --is_train --is_crop False --epoch 10 --dataset2 test_cat_G
    '''
