import os
import glob
import numpy as np
import shutil
import json
from tqdm import tqdm

"""
Creates dummy audio files if there is no feature.
"""

# new generic ds format

ds_base_path = '/path/to/Datasets/VATEX/'
json_ds_info = os.path.join(ds_base_path, 'dataset_info.json')
with open(json_ds_info, 'r') as f:
    ds_info = json.load(f)


feat_path = os.path.join(ds_base_path, 'features_ResNet101V2')
audio_feats_path = os.path.join(ds_base_path, 'features_audio') 
audio_feats_path_enhanced = os.path.join(ds_base_path, 'features_audio_enhanced')
# create dir if not exists
os.makedirs(audio_feats_path_enhanced, exist_ok=True)

all_i3d_paths = glob.glob(os.path.join(feat_path, '*.feat'))

def get_audio_feature_path(basename, kind='raw'):
    basename = os.path.splitext(basename)[0]
    audio_fname = os.path.join(audio_feats_path, f'{basename}_vggish_{kind}.bin')
    return audio_fname

def generate_dummy_feature(dst_path):
    dummy_feat = np.zeros((1, 128), dtype=np.float32)
    dummy_feat.tofile(dst_path)

for i3d_file in tqdm(all_i3d_paths):
    basename = os.path.basename(i3d_file)
    audio_feat_path = get_audio_feature_path(basename)
    audio_basename = os.path.basename(audio_feat_path)
    dst_path = os.path.join(audio_feats_path_enhanced, audio_basename)
    if not os.path.exists(audio_feat_path):
        print(f"{basename} audio feature does not exist. Creating dummy --> {dst_path}")
        generate_dummy_feature(dst_path)
    else:
        audio_fstat = os.stat(audio_feat_path)
        if audio_fstat.st_size > 0:
            shutil.copyfile(audio_feat_path, dst_path)
        else:
            print(f"{basename} audio feature empty (size==0). Creating dummy --> {dst_path}")
            generate_dummy_feature(dst_path)