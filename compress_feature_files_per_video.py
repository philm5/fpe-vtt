import json
import os
import natsort
from tqdm import tqdm
import numpy as np
import glob

"""
Compresses single frame video features into a single file. E.g., features [1.jpg.feat, ..., 300.jpg.feat] --> video_id.feat
"""

base_path = '/dataset/base/path/'

json_path = os.path.join(base_path, 'dataset_info.json')
feat_path = os.path.join(base_path, 'features_ResNet101V2')

print("Loading dataset json from disk...")
with open(json_path, 'r') as f:
    dataset_info = json.load(f)

videos = dataset_info['videos']

srt = natsort.natsort_keygen(key=lambda y: y.lower())
ntkeygen = natsort.natsort_keygen(key=lambda y: y.lower())

for sample in tqdm(videos):
    video_frames_path = os.path.join(feat_path, sample['video_id'])
    video_frames_feature_files = glob.glob(os.path.join(video_frames_path, '*.feat'))

    video_frames_feature_files.sort(key=ntkeygen)
    feats = []
    for feat in video_frames_feature_files:
        feat_data = np.fromfile(feat, dtype=np.float32)
        feats.append(feat_data)

    feats_ndarr = np.asarray(feats)
    full_feat_path = os.path.join(feat_path, f"{sample['video_id']}.feat")

    feats_ndarr.tofile(full_feat_path)