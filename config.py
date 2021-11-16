import os
import tensorflow as tf

# feat_path_cnn_resnet_101_v2 = /dataset/base/path/features_ResNet101V2'
# feat_path_i3d = /dataset/base/path/features_i3d_final'
# feat_path_i3d_spatial = /dataset/base/path/features_i3d_final_spatial'
# feat_path_i3d_spatial_meantime = /dataset/base/path/features_i3d_final_spatial_meantime'
# audio_feats_path = /dataset/base/path/features_audio_enhanced/'
feat_path_cnn_resnet_101_v2 = 'features_ResNet101V2'
feat_path_simclr_r50_2x_sk1 = 'features_clr_r50_2x_sk1'
feat_path_simclr_r101_2x_sk1 = 'features_clr_r101_2x_sk1'
feat_path_i3d = 'features_i3d'
feat_path_i3d_spatial = 'features_i3d_spatial'
features_i3d_spatial_rgb_imagenet = 'features_i3d_spatial_rgb_imagenet'


feat_path_i3d_spatial_meantime = 'features_i3d_spatial_meantime'
audio_feats_path = 'features_audio_enhanced'
raw_image_path = 'frames'

#base_paths, video_ids

feat_preproc_dict = \
{
    'cnn_resnet_101_v2':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_cnn_resnet_101_v2, 'feat'),
            'shape': [-1, 2048],
            'out_shape': [-1, 2048],
        },
    'simclr_r50_2x_sk1':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_simclr_r50_2x_sk1, 'feat'),
            'shape': [-1, 2048],
            'out_shape': [-1, 2048],
        },
    'simclr_r101_2x_sk1':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_simclr_r101_2x_sk1, 'feat'),
            'shape': [-1, 2048],
            'out_shape': [-1, 2048],
        },
    'i3d':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_i3d, 'bin'),
            'shape': [-1, 1024],
            'out_shape': [-1, 1024],
        },
    'i3d_spatial':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_i3d_spatial, 'bin'),
            'shape': [-1, 7, 7, 1024],
            'out_shape': [-1, 7, 7, 1024],
        },
    'features_i3d_spatial_rgb_imagenet':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, features_i3d_spatial_rgb_imagenet, 'bin'),
            'shape': [-1, 7, 7, 1024],
            'out_shape': [-1, 7, 7, 1024],
        },
    'i3d_spatial_meantime':
        {
            'get_feature_paths': lambda base_path, video_id: get_feature_path(base_path, video_id, feat_path_i3d_spatial_meantime, 'bin'),
            'shape': [7, 7, 1024],
            'out_shape': [7, 7, 1024],
        },

    'image_raw':
        {
            'get_feature_paths': lambda base_path, video_id, num_frames: get_raw_img_wildcard(base_path, video_id, raw_image_path, num_frames),
            'shape': [-1, 224, 224, 3],
            'out_shape': [-1, 224, 224, 3],
        },
    'audio_raw':
        {
            'get_feature_paths': lambda base_path, video_id: get_audio_feature_path(base_path, video_id, audio_feats_path, 'raw'),
            'shape': [-1, 128],
            'out_shape': [128],
        },
}


def transform_shape(shp):
    # replaces -1 with None
    return [None if x == -1 else x for x in shp]

def transform_shape(shp):
    # replaces -1 with None
    return [None if x == -1 else x for x in shp]

def get_audio_feature_path(base_path, video_id, audio_feats_path, kind='raw'):
    full_feat_path = os.path.join(base_path, audio_feats_path, f"{video_id}_vggish_{kind}.bin")
    return full_feat_path

def get_feature_path(base_path, video_id, feat_base_path, ext):
    full_feat_path = os.path.join(base_path, feat_base_path, f"{video_id}.{ext}")
    return full_feat_path

def get_raw_img_wildcard(base_path, video_id, feat_base_path, num_frames):
    full_feat_paths = [os.path.join(base_path, feat_base_path, video_id, f"{x+1}.jpg") for x in range(num_frames)]
    return full_feat_paths