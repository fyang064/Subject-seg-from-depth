import os
import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import argparse

class PersonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "humanbody"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + person

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")   
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs") 
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect persons.')
parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
args = parser.parse_args()
                        
run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    #net = MobileNet(alpha=.75, input_tensor=tf.placeholder('float32', shape=(1,32,32,3)))
    config = PersonConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # Return number of parameters and flops
    print("FLOPS: {:,} --- Params: {:,}".format(flops.total_float_ops, params.total_parameters))
