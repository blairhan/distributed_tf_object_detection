## modify model_lib.py additionally
import functools
import json
import os
import tensorflow as tf

import model_lib
import inputs
from object_detection import model_hparams
from object_detection.builders import model_builder
from object_detection.utils import config_util

## ADDED for multi-gpu
import horovod.tensorflow as hvd



flags = tf.app.flags

tf.app.flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('train_dir', '/root/train/test', 'Training dataset')
flags.DEFINE_string('pipeline_config_path', '/root/train/test/test.config', 'Training dataset')

FLAGS = flags.FLAGS



def main(unused_argv):
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    hvd.init()

    # Configurations
    
    ## ADDED for multi-gpu
    sconfig = tf.ConfigProto()
    sconfig.gpu_options.allow_growth = True
    sconfig.gpu_options.visible_device_list = str(hvd.local_rank())
    model_dir = FLAGS.train_dir if hvd.rank() == 0 else None
    
    hparam = model_hparams.create_hparams(None)
    run_config = tf.estimator.RunConfig(model_dir= model_dir, session_config= sconfig, save_checkpoints_secs=1000)
    configs = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path, config_override=None)
    configs = config_util.merge_external_params_with_configs(configs, hparams=hparam, kwargs_dict={})
    
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    eval_input_config = configs['eval_input_config']
    eval_config = configs['eval_config']

    
    # Prepare Functions
    train_input_fn = inputs.create_train_input_fn(train_config = train_config, train_input_config = train_input_config, model_config = model_config)
    detection_model_fn = functools.partial(model_builder.build, model_config = model_config, is_training = False)
    model_fn = model_lib.create_model_fn(detection_model_fn, configs, hparams = hparam, use_tpu = False)

    ## ADDED for multi-gpu
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    

    
    estimator.train(input_fn=train_input_fn, steps=train_config.num_steps, hooks=[bcast_hook])


if __name__ == '__main__':
  tf.app.run()