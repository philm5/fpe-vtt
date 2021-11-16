import tensorflow as tf
import dataset_transformer
import os
import sys
import json
from calc_scores import calculate_scores, init_coco
from subprocess import call
from config import *
from pathlib import Path
from vtt_transformer import *
import time
from transformer_tools import TransformerTools
from tqdm import tqdm, trange
from tensorflow.keras.mixed_precision import experimental as mixed_precision

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

val_gt_fname = 'val_gt.json'
val_samples_fname = 'val_samples.json'

GLOBAL_BATCH_SIZE = 32

def start_eval(ckpt_path, val_path):
    # Init transformer tools
    transformer_tools = TransformerTools(
        params=params,
        dist_train_dataset=None,
        dist_eval_dataset=dist_eval_dataset,
        strategy=strategy,
        coco=coco,
        mode=tf.estimator.ModeKeys.TRAIN,
        model_type=params['model_type']
    )

    print("Trying to restore from checkpoint...")
    sys.stdout.flush()
    transformer_tools.try_init_from_checkpoint(ckpt_path)
    sys.stdout.flush()


    @tf.function(input_signature=[dist_eval_dataset.element_spec])  # experimental_relax_shapes=True
    def distributed_inference_step(dist_inputs):
        return strategy.run(transformer_tools.predict, args=(dist_inputs,))


    # ===================================================================================================================
    # Actual test loop -------------------------------------------------------------------------------------------
    # ===================================================================================================================
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(params['train_dir'], 'test'))

    num_eval_steps = num_eval_samples // GLOBAL_BATCH_SIZE

    val_ds_name = os.path.basename(os.path.normpath(val_path))

    with eval_summary_writer.as_default():
        transformer_tools.eval_performance(-2, num_eval_steps,
                                           evalstep=distributed_inference_step,
                                           test=True,
                                           custom_name=f"test_{val_ds_name}")


if __name__ == "__main__":
    last_exp = "/path/to/params/json/params.json"

    with open(last_exp, 'r') as f:
        params = json.load(f)

    print(json.dumps(params, indent=4, sort_keys=True))

    params['batch_size'] = GLOBAL_BATCH_SIZE


    val_path = '/path/to/dataset/with/splits/'

    with strategy.scope():
        ds_params = params
        ds_params['max_num_frames'] = 10000
        ds_params['different_val_samples_fname'] = val_samples_fname

        ds_eval, num_eval_samples = dataset_transformer.get_dataset(dataset_path=val_path,
                                                                    mode=tf.estimator.ModeKeys.EVAL,
                                                                    params=ds_params,
                                                                    batch_size=GLOBAL_BATCH_SIZE)
        data_options = tf.data.Options()
        data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ds_eval = ds_eval.with_options(data_options)

    dist_eval_dataset = strategy.experimental_distribute_dataset(ds_eval)

    # Init coco
    coco = init_coco(os.path.join(val_path, val_gt_fname))


    checkpoint_dir = params['train_dir']

    ckpt_path = "./checkpoints/MODEL_NO/best_model/ckpt-58000"

    start_eval(ckpt_path, val_path) # params)