import tensorflow as tf
import os
import pathlib
import numpy as np
from dataset_feature_extraction import get_dataset
from feature_extractor_model import FeatureExtractor
from tqdm import tqdm

"""
Extracts basic ResNet101V2 features for a dataset.
"""
base_path = '/dataset/base/path/'

feat_path = os.path.join(base_path, 'features_ResNet101V2')

model = FeatureExtractor()
ds, num_batches = get_dataset(ds_base_path=base_path)


@tf.function
def forward(input):
    features = model(input)
    return features

for img, filename in tqdm(ds, total=num_batches):
    features = forward(img)
    np_feats = features.numpy()
    fnames = filename.numpy()

    for fname, np_feat in zip(fnames, np_feats):
        fn = fname.decode('utf-8')
        basedir = os.path.basename(os.path.dirname(fn))
        basename = os.path.basename(fn)
        full_feat_path = os.path.join(feat_path, basedir, f"{basename}.feat")

        dstdir = os.path.join(feat_path, basedir)
        pathlib.Path(dstdir).mkdir(parents=True, exist_ok=True)

        np_feat.tofile(full_feat_path)

