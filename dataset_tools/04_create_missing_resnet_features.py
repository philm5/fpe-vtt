import os
import glob
import numpy as np
import shutil
import json
from tqdm import tqdm


# new generic ds format

ds_base_path = '/path/to/Datasets/VATEX_test_reduced_simulated'
json_ds_info = os.path.join(ds_base_path, 'dataset_info.json')
with open(json_ds_info, 'r') as f:
    ds_info = json.load(f)

feat_path = os.path.join(ds_base_path, 'features_i3d_spatial_rgb_imagenet') 
resnet_feats_path = os.path.join(ds_base_path, 'features_ResNet101V2') 

all_i3d_paths = glob.glob(os.path.join(feat_path, '*.bin'))

def get_resnet_feature_path(basename, kind='raw'):
    basename = os.path.splitext(basename)[0]
    audio_fname = os.path.join(resnet_feats_path, f'{basename}.feat')
    return audio_fname

def generate_dummy_feature(dst_path):
    dummy_feat = np.zeros((1, 2048), dtype=np.float32)
    dummy_feat.tofile(dst_path)

for i3d_file in tqdm(all_i3d_paths):
    basename = os.path.basename(i3d_file)
    audio_feat_path = get_resnet_feature_path(basename)
    dst_path = audio_feat_path
    if not os.path.exists(audio_feat_path):
        print(f"{basename} resnet feature does not exist. Creating dummy --> {dst_path}")
        generate_dummy_feature(dst_path)