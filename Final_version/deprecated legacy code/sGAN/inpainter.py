import os
import sys
import tensorflow as tf
from model import DCGAN

# Define a tensorflow app and the flags
flags = tf.app.flags

flags.DEFINE_integer(flag_name="nIter", default_value=5000,
                     docstring="Iterations in total [5000]")

flags.DEFINE_integer(flag_name="input_height", default_value=64,
                     docstring="The size of image to use (will be center cropped). [108]")

flags.DEFINE_integer(flag_name="input_width", default_value=None,
                     docstring="The size of image to use (will be center cropped). If None, same value as input_height [None]")

flags.DEFINE_float(flag_name="lr", default_value=0.01,
                   docstring="Learning rate [0.5]")

flags.DEFINE_float(flag_name="momentum", default_value=0.8,
                   docstring="Momentum term of AdamOptimizer [0.8]")

flags.DEFINE_float(flag_name="gpu_utilization", default_value=0.8,
                   docstring="Per process GPU memory fraction [0.8]")

flags.DEFINE_float(flag_name="lambda_val", default_value=0.08,
                   docstring="Lambda value for contextual and percptual loss [0.08]")

flags.DEFINE_string(flag_name="checkpointDir", default_value="checkpoint",
                    docstring="Directory name to save/load the checkpoints [checkpoint]")

flags.DEFINE_string(flag_name="outDir", default_value="inpainted",
                    docstring="Directory name to output inpainting results [inpaintings]")

flags.DEFINE_string(flag_name="maskType", default_value="random",
                    docstring="The type of masks, four choices:{'random', 'center', 'left', 'full'} [center] ")

flags.DEFINE_string(flag_name="imgs", default_value="testing_data",
                    docstring="Directory name to testing data that will be painted [testing_data]")

flags.DEFINE_integer(flag_name="c_dim", default_value=3,
                     docstring="Dimension of image color/channels. [3]")

flags.DEFINE_string(flag_name="dataset", default_value="celebA",
                    docstring="The name of dataset [celebA, mnist, lsun]")

flags.DEFINE_string(flag_name="input_file_extension", default_value="*.jpg",
                    docstring="Glob pattern of filename of input images [*]")

flags.DEFINE_boolean(flag_name="is_crop", default_value=False,
                     docstring="True for training, False for testing [False]")


FLAGS = flags.FLAGS


def main(_):
    assert(os.path.exists(FLAGS.checkpointDir))

    # Deal with input size default values
    if not FLAGS.input_width:
        FLAGS.input_width = FLAGS.input_height

    def get_default_gpu_session(fraction=0.8):
        if fraction > 1:
            fraction = 0.8
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        return tf.Session(config=config)

    with get_default_gpu_session(FLAGS.gpu_utilization) as sess:

        dcgan_instance = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            c_dim=FLAGS.c_dim,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_file_extension,
            is_crop=FLAGS.is_crop,
            checkpoint_dir=FLAGS.checkpointDir,
            lambda_val=FLAGS.lambda_val
        )
        dcgan_instance.inpaint(FLAGS)

if __name__ == '__main__':
    tf.app.run()
    '''
    python inpainter.py --nIter 10000 --is_crop --dataset catface --imgs /media/rex/Extra/DeepWorking/ccGAN/XGAN/to_test/ --outDir /media/rex/Extra/DeepWorking/ccGAN/XGAN/Inpating --maskType center
    
    python inpainter.py --dataset mnist_train --imgs /media/rex/Extra/DeepWorking/ccGAN/data/mnist_test/ --outDir /media/rex/Extra/DeepWorking/ccGAN/Inpating --is_train --nIter 1000 --is_crop --input_height=28 --output_height=28 --c_dim 1 --maskType random

    '''
