import tensorflow as tf
import dataset_transformer
import os
import sys
import json
from calc_scores import calculate_scores, init_coco
from subprocess import call
from config import *
from vtt_transformer import *
import time
from transformer_tools import TransformerTools
from tqdm import tqdm, trange
from tensorflow.keras.mixed_precision import experimental as mixed_precision

""" Code similar to train_transformer.py but with REINFORCE, i.e., Self-critical sequence learning"""

# Limit growing GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)


DATASET_PATH = '/path/to/dataset/'

params = {
    'model_type': "default",
    'network': "transformer",
    'd_model': 512,
    'num_layers': 8,
    'num_heads': 8,
    'dff': 2048,
    'memory_vector_size': 64,
    'pe_input': 2000,
    'pe_target': 2000,
    'num_epochs': 200,
    'transformer_warmup_steps': 10000,
    'label_smoothing': 0.1,
    'lr_schedule': "cosine_warmup",
    'use_timestamp_factors': True,
    'batch_size': None,
    'vocab_size': 12000,
    'videobert_size': 10000 + 100, 
    'img_feats': 'cnn_resnet_101_v2',
    'i3d_feats': 'features_i3d_spatial_rgb_imagenet',
    'use_i3d_instead_of_imgs': True,  # !
    'use_aud_feats': True,
    'max_num_frames': 300,

    'caption_id_feature_name': "caption_ids",  # 'video_features'
    'caption_mask_feature_name': "mask",  # 'video_features'
    'frame_mask_feature_name': "num_frames_mask",  # 'num_i3d_frames_mask', 'num_frames_mask'
    'image_feature_name': 'features',  # 'i3d_features',  'features'
    'i3d_feature_name': 'i3d_features',
    'i3d_frame_mask_feature_name': 'num_i3d_frames_mask',
    'aud_feats': 'audio_raw',
    'audio_feature_name': 'aud_features',

    'use_subword_vocab': 'BERT',
    'finetune': {
        "init_from": "./checkpoints/train/MODEL_NAME/best_model/ckpt-20817",
        'train_ds_json': None,
    },
    'use_augmentor_network': None, 
    'init_word_embedding':  None, 
    'init_video_encoder': None,
    'init_videobert': None, 
}

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BATCH_SIZE_PER_REPLICA = 6 # !
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
params['batch_size'] = GLOBAL_BATCH_SIZE

if __name__ == "__main__":
    eval = True
    coco = init_coco(os.path.join(DATASET_PATH, 'val_gt.json'))

    params['dataset_path'] = DATASET_PATH

    params['datasplit'] = os.path.basename(os.path.normpath(DATASET_PATH))  # videodatainfo['info']['name']


    with open('./last_experiment.txt', 'r') as f:
        model_no = int(f.readline().strip()) + 1

    if params['model_type'] == 'LSTM':
        params['network'] = 'LSTM'
    else:
        params['network'] = 'transformer'

    model_suffix = ''
    if 'finetune' in params:
        if not params['finetune'] is None:
            model_suffix = '_finetune'

    params['MODEL_NAME'] = "{}_{}_{}{}".format(model_no, params['datasplit'], params['network'], model_suffix)
    params[
        'train_dir'] = f"./checkpoints/train/{params['MODEL_NAME']}/"  # TRAILING / required!'./train/{}'.format(params['MODEL_NAME'])

    os.makedirs(params['train_dir'], exist_ok=True)


    # Create Datasets --------------------------------------------------------------------------------------------------
    if not params['finetune'] is None and 'train_ds_json' in params['finetune'] and not params['finetune']['train_ds_json'] is None:
            train_path = params['finetune']['train_ds_json']
            val_path = params['finetune']['val_ds_json']
    else:
        train_path = params['dataset_path']
        val_path = params['dataset_path']


    with strategy.scope():
        ds_train, num_train_samples = dataset_transformer.get_dataset(dataset_path=train_path,
                                                                      mode=tf.estimator.ModeKeys.TRAIN,
                                                                      params=params,
                                                                      batch_size=GLOBAL_BATCH_SIZE)
        if eval:
            ds_eval, num_eval_samples = dataset_transformer.get_dataset(dataset_path=val_path,
                                                                        mode=tf.estimator.ModeKeys.EVAL,
                                                                        params=params,
                                                                        batch_size=GLOBAL_BATCH_SIZE)

        data_options = tf.data.Options()
        data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ds_train = ds_train.with_options(data_options)
        if eval:
            ds_eval = ds_eval.with_options(data_options)

    dist_train_dataset = strategy.experimental_distribute_dataset(ds_train)
    if eval:
        dist_eval_dataset = strategy.experimental_distribute_dataset(ds_eval)

    # Calculate training / eval steps
    train_steps_per_epoch = num_train_samples // GLOBAL_BATCH_SIZE
    params['num_samples_per_epoch'] = num_train_samples
    params['train_steps_per_epoch'] = train_steps_per_epoch
    if eval:
        num_eval_steps = num_eval_samples // GLOBAL_BATCH_SIZE
    else:
        dist_eval_dataset = None

    # Init transformer tools
    transformer_tools = TransformerTools(
        params=params,
        dist_train_dataset=dist_train_dataset,
        dist_eval_dataset=dist_eval_dataset,
        strategy=strategy,
        coco=coco,
        mode=tf.estimator.ModeKeys.TRAIN,
        model_type=params['model_type'] #=False
    )
    # Create vocabulary info for Tensorboard ---------------------------------------------------
    transformer_tools.create_tensorboard_vocab_info()

    with open('./params.json', 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    # also save old params into params folder
    with open('./params/params_{}.json'.format(params['MODEL_NAME']), 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    init_from = None
    if not params['finetune'] is None:
        if 'init_from' in params['finetune']:
            init_from = params['finetune']['init_from']

    print("Trying to restore from checkpoint...")
    sys.stdout.flush()
    transformer_tools.try_init_from_checkpoint(init_from)
    sys.stdout.flush()
    #print('dense1.kernel.dtype: %s' % transformer_tools.model.final_layer.dtype)



    @tf.function(input_signature=[dist_train_dataset.element_spec])  # experimental_relax_shapes=True
    def distributed_train_step(dist_inputs):

        #per_replica_losses = strategy.run(transformer_tools.train_step, args=(dist_inputs,))
        per_replica_losses = strategy.run(transformer_tools.train_step_rl, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    if eval:
        @tf.function(input_signature=[dist_eval_dataset.element_spec])  # experimental_relax_shapes=True
        def distributed_inference_step(dist_inputs):
            return strategy.run(transformer_tools.predict, args=(dist_inputs,))


    # ===================================================================================================================
    # Actual train/test loop -------------------------------------------------------------------------------------------
    # ===================================================================================================================
    if eval:
        eval_summary_writer = tf.summary.create_file_writer(os.path.join(params['train_dir'], 'eval'))
    train_summary_writer = tf.summary.create_file_writer(os.path.join(params['train_dir']))


    #
    #transformer_tools.model.encoder.trainable = False

    train = True
    step = 0
    for e in range(params['num_epochs']):
        if train:
            batch = 0
            total_loss = 0.0
            start = time.time()

            transformer_tools.train_accuracy.reset_states()
            
            with train_summary_writer.as_default():
                t = tqdm(dist_train_dataset, total=train_steps_per_epoch)
                for dist_inputs in t:

                    transformer_tools.global_norm.reset_states()
                    transformer_tools.train_loss.reset_states()
                    transformer_tools.avg_reward.reset_states()
                    transformer_tools.avg_train_ciderd.reset_states()
                    transformer_tools.greedy_sample_sentence_length.reset_states()
                    transformer_tools.train_sample_sentence_length.reset_states()

                    start = time.time()
                    loss = distributed_train_step(dist_inputs)
                    end = time.time()

                    step += 1
                    batch += 1

                    report_iter = transformer_tools.optimizer.iterations.numpy()
                    if report_iter % 500 == 0:
                        transformer_tools.save_checkpoint(report_iter)
                    tf.summary.scalar('stats/time_per_step', end - start, step=report_iter)
                    tf.summary.scalar('lr', transformer_tools.optimizer._decayed_lr('float32').numpy(),
                                      step=report_iter)
                    for metric_obj in transformer_tools.train_metrics():
                        tf.compat.v2.summary.scalar(metric_obj.name, metric_obj.result(),
                                                    report_iter)
                        #train_summary_writer.flush()
                    t.set_postfix({
                         'l': loss.numpy(),
                         'acc': transformer_tools.train_accuracy.result().numpy(),
                         'r': transformer_tools.avg_reward.result().numpy(),
                         'e': e + 1
                    })

            transformer_tools.save_checkpoint(report_iter)
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(e + 1,
                                                                transformer_tools.train_loss.result(),
                                                                transformer_tools.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            if e == 1:
                # actually increment experiment number if at least one epoch was trained.
                with open('./last_experiment.txt', 'w') as f:
                    f.write(str(model_no))
        if eval:
            with eval_summary_writer.as_default():
                transformer_tools.eval_performance(e, num_eval_steps, evalstep=distributed_inference_step)