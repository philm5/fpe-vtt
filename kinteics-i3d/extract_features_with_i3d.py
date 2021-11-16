
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import i3d
import sonnet as snt
import os
import glob
from tqdm import tqdm
from dataset_feature_extraction import get_dataset_feat_ext
from pathlib import Path

"""
Adapted code in order to extract I3D features. Original code in this folder from: https://github.com/deepmind/kinetics-i3d
"""


_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 79
NUM_CLASSES = 400

# one of:
# features_i3d_final_spatial_meantime            --> shape: [7, 7, 1024]
# features_i3d_spatial                           --> shape: [-1, 7, 7, 1024]
# features_i3d                                   --> shape: [-1, 1024]
mode = 'features_i3d_spatial' # features_i3d_spatial_rgb_imagenet
net = 'rgb_imagenet'
ds_base_path = '/path/to/Dataset/base/path/VATEX'

ds, num_samples = get_dataset_feat_ext(ds_base_path, video_segment_len=None, fps=None) #, video_segment_len=1.5, fps=20)

feat_path = os.path.join(ds_base_path, f"{mode}_{net}")

Path(feat_path).mkdir(parents=True, exist_ok=True)

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

def main(unused_argv):
    # RGB input has 3 channels.

    it = ds.make_one_shot_iterator()
    # it = ds.make_one_shot_iterator()
    # next = it.get_next()

    vname, split, rgb_input = it.get_next()
    #rgb_input = tf.Print(rgb_input, [tf.shape(rgb_input)], summarize=10)

    #rgb_input = tf.squeeze(rgb_input) # [224 224 3]
    #rgb_input = tf.expand_dims(rgb_input, axis=0) # [1 224 224 3]
    #rgb_input = tf.Print(rgb_input, [rgb_input, tf.shape(rgb_input)], message='RGB',  summarize=20)
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, all_endpoints = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    my_feature_map = all_endpoints['Mixed_5c'] # Results in i3d_spatial feats, i.e., [1, -1, 7, 7, 1024]
    #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map), my_feature_map],  summarize=20, message='mfm')

    if mode == 'features_i3d_final_spatial_meantime' or mode == 'features_i3d':
        # Keep spatial dims, mean over time dim.

        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map)], summarize=20, message='i3d_0')
        my_feature_map = tf.nn.avg_pool3d(my_feature_map, ksize=[1, 2, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='VALID')
        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map)], summarize=20, message='i3d_1')
        my_feature_map = tf.reduce_mean(my_feature_map, axis=1)
        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map)], summarize=20, message='i3d_2')
        my_feature_map = tf.squeeze(my_feature_map) # shape: [7, 7, 1024]
        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map)], summarize=20, message='i3d_3')

    if mode == 'features_i3d':
        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map)], summarize=20, message='i3d_4')
        my_feature_map = tf.reduce_mean(my_feature_map, axis=[0, 1])
        #my_feature_map = tf.Print(my_feature_map, [tf.shape(my_feature_map), my_feature_map], summarize=20, message='mfm_postproc')

    #my_feature_map = tf.squeeze(my_feature_map, [2, 3], name='SpatialSqueeze')
    # Like in i3d
    # my_feature_map = tf.nn.avg_pool3d(my_feature_map, ksize=[1, 2, 7, 7, 1],
    #                       strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    # my_feature_map = tf.squeeze(my_feature_map, [2, 3], name='SpatialSqueeze')


        #net = tf.reduce_mean(net, axis=1)

    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
          rgb_variable_map[variable.name.replace(':0', '')] = variable


    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[net])
        tf.logging.info('RGB checkpoint restored')

        # it = dataset.make_one_shot_iterator()
        # next = it.get_next()
        for _ in tqdm(range(num_samples)):
            #a = sess.run(rgb_input)
            out_fname, out_split, out_map = sess.run(
                [vname, split, my_feature_map])

            #print(out_map[0].shape
            if out_split == -1:
                out_map.tofile(os.path.join(feat_path, f'{out_fname.decode()}.bin'))
            else:
                out_map.tofile(os.path.join(feat_path, f'{out_fname.decode()}-{out_split}.bin'))


    #print(out_map[0].shape)
if __name__ == '__main__':
    tf.app.run(main)