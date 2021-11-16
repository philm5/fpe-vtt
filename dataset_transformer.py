import tensorflow as tf
import pandas as pd
import random
import json
import os
from config import *
import sys

def my_transform(sample, key):
    if key in sample:
        return sample[key]
    else:
        return [0]

def get_lists_for_dataset(mode, dataset_path, params):

    #if params['use_raw_images']:
    #    preproc_img_raw = feat_preproc_dict['image_raw']
    preproc_img = feat_preproc_dict[params['img_feats']]
    preproc_i3d = feat_preproc_dict[params['i3d_feats']]
    preproc_aud = feat_preproc_dict[params['aud_feats']]

    if os.path.isfile(dataset_path):
        json_path = dataset_path
    elif os.path.isdir(dataset_path):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if params['model_type'] == "videobert":
                json_path = os.path.join(dataset_path, 'train_samples_videobert.json')
            else:
                json_path = os.path.join(dataset_path, 'train_samples.json')
        else:
            if params['model_type'] == "videobert":
                json_path = os.path.join(dataset_path, 'val_samples_videobert.json')
            else:
                val_samples_fname = 'val_samples.json'
                if 'different_val_samples_fname' in params:
                    val_samples_fname = params['different_val_samples_fname']
                json_path = os.path.join(dataset_path, val_samples_fname)


    print("Loading dataset json from disk...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # Only use every video once in eval mode!
    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
        uniqList = {x['video_id']: x for x in json_data}
        uniqList = uniqList.values()
        json_data = [x for x in uniqList]

    json_data = [x for x in json_data if 10 < x['num_frames'] <= params['max_num_frames']]

    video_ids = list(map(lambda x: my_transform(x, 'video_id'), json_data))
    sen_ids = list(map(lambda x: my_transform(x, 'sen_id'), json_data))
    base_paths = list(map(lambda x: my_transform(x, 'base_path'), json_data))
    num_frames = list(map(lambda x: my_transform(x, 'num_frames'), json_data))

    i3d_timestamp_factor = list(map(lambda x: my_transform(x, 'video_infos_i3d_timestamp_factor'), json_data))
    aud_timestamp_factor = list(map(lambda x: my_transform(x, 'video_infos_aud_timestamp_factor'), json_data))

    base_paths_and_video_ids = list(zip(base_paths, video_ids))
    base_paths_and_video_ids_num_frames = list(zip(base_paths, video_ids, num_frames))

    images_feature_path = [preproc_img['get_feature_paths'](x[0], x[1]) for x in base_paths_and_video_ids]
    i3d_feature_path = [preproc_i3d['get_feature_paths'](x[0], x[1]) for x in base_paths_and_video_ids]
    audio_feature_path = [preproc_aud['get_feature_paths'](x[0], x[1]) for x in base_paths_and_video_ids]

    # full_feat_path = os.path.join(feat_path, f"{basedir}.feat")
    captions = list(map(lambda x: my_transform(x, 'caption'), json_data))
    captions_ids = list(map(lambda x: my_transform(x, 'caption_ids'), json_data))
    videobert_tokens = list(map(lambda x: my_transform(x, 'videobert_tokens'), json_data))

    return json_data, \
           video_ids, \
           sen_ids, \
           base_paths, \
           base_paths_and_video_ids, \
           images_feature_path, \
           i3d_feature_path, \
           audio_feature_path, \
           captions, \
           captions_ids, \
           videobert_tokens, \
           i3d_timestamp_factor, \
           aud_timestamp_factor

def get_generator_transformer(dataset_lists):
    json_data, video_ids, sen_ids, base_paths, \
    base_paths_and_video_ids, images_feature_path, \
    i3d_feature_path, audio_feature_path, captions, \
    captions_ids, videobert_tokens,  i3d_timestamp_factor, \
    aud_timestamp_factor = dataset_lists

    def my_generator(json_data, video_ids, sen_ids, images_feature_path,
                     i3d_feature_path, audio_feature_path, captions_ids,
                     captions, videobert_tokens, i3d_timestamp_factor,
                     aud_timestamp_factor):
        for idx in range(len(json_data)):
            yield (video_ids[idx],
                   sen_ids[idx],
                   images_feature_path[idx],
                   i3d_feature_path[idx],
                   audio_feature_path[idx],
                   captions_ids[idx],
                   captions[idx],
                   videobert_tokens[idx],
                   i3d_timestamp_factor[idx],
                   aud_timestamp_factor[idx])

    gen = lambda: my_generator(json_data=json_data,
                       video_ids=video_ids,
                       sen_ids=sen_ids,
                       images_feature_path=images_feature_path,
                       i3d_feature_path=i3d_feature_path,
                       audio_feature_path=audio_feature_path,
                       captions_ids=captions_ids,
                       captions=captions,
                       videobert_tokens=videobert_tokens,
                       i3d_timestamp_factor=i3d_timestamp_factor,
                       aud_timestamp_factor=aud_timestamp_factor)

    return gen

def get_dataset(dataset_path, mode, params, batch_size=32, shuffle=False):
    preproc_img = feat_preproc_dict[params['img_feats']]
    preproc_i3d = feat_preproc_dict[params['i3d_feats']]
    preproc_aud = feat_preproc_dict[params['aud_feats']]

    preproc_img_raw = feat_preproc_dict['image_raw']

    dataset_lists = get_lists_for_dataset(mode, dataset_path, params)
    json_data = dataset_lists[0]

    generator = get_generator_transformer(dataset_lists)

    output_types = (tf.string, tf.string, tf.string, tf.string, tf.string, tf.int32, tf.string, tf.int32, tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([None]),
        tf.TensorShape([]),
        tf.TensorShape([None]),
        tf.TensorShape([]),
        tf.TensorShape([]),
    )


    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types, output_shapes=output_shapes)


    if mode == tf.estimator.ModeKeys.TRAIN or shuffle:
        dataset = dataset.shuffle(buffer_size=len(json_data))

    def load_features_from_disk(path):
        file = tf.io.read_file(path)
        feats = tf.io.decode_raw(file, out_type=tf.float32)
        return feats

    def read_image(path, mode):
        image_raw = tf.io.read_file(path)
        image_decoded = tf.image.decode_png(image_raw, channels=3)
        image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

        if mode == tf.estimator.ModeKeys.TRAIN:
            image_decoded = tf.image.resize(image_decoded, [256, 256])
            image_decoded = tf.image.random_crop(image_decoded, [224, 224, 3])
            image_decoded = tf.image.random_flip_left_right(image_decoded)
        else:
            image_decoded = tf.image.resize(image_decoded, [224, 224])

        image_decoded = tf.subtract(image_decoded, 0.5)
        image_decoded = tf.multiply(image_decoded, 2.0)
        return image_decoded

    def load_preproc_feature(feature_path, shape):
        # load i3d frame features / no spatial dim with shape [NUM_FRAMES, 1024] from disk
        video_feats = load_features_from_disk(feature_path)

        # 1024 for i3d features, 2048 for default features from per image extraction of resnet
        video_feats = tf.reshape(video_feats, shape)
        return video_feats

    def load_images(image_paths, num_frames, shape):
        # load raw images and resize them
        raw_imgs = tf.map_fn(lambda x: read_image(x, tf.estimator.ModeKeys.EVAL), elems=image_paths, fn_output_signature=tf.float32)

        preproc_raw_imgs = tf.reshape(raw_imgs, shape)
        return preproc_raw_imgs


    def prepare_sample(video_id,
                       sen_id,
                       images_feature_path,
                       i3d_feature_path,
                       aud_feature_path,
                       caption_ids,
                       caption,
                       videobert_tokens,
                       i3d_timestamp_factor,
                       aud_timestamp_factor):

        images_shape = preproc_img['shape']
        images_feats = load_preproc_feature(images_feature_path, images_shape)

        i3d_shape = preproc_i3d['shape']
        #i3d_feats = tf.zeros(shape=(1, 1024), dtype=tf.float32)
        i3d_feats = load_preproc_feature(i3d_feature_path, i3d_shape)


        aud_shape = preproc_aud['shape']
        aud_feats = load_preproc_feature(aud_feature_path, aud_shape)

        frames_vid = tf.shape(images_feats)[0]
        num_frames_mask = tf.ones(dtype=tf.float32, shape=frames_vid)
        i3d_vid = tf.shape(i3d_feats)[0]
        num_i3d_frames_mask = tf.ones(dtype=tf.float32, shape=i3d_vid)
        aud_num_frames = tf.shape(aud_feats)[0]
        num_aud_frames_mask = tf.ones(dtype=tf.float32, shape=aud_num_frames)

        caption_len = tf.size(caption_ids)
        mask = tf.ones(dtype=tf.float32, shape=caption_len)

        videobert_len = tf.size(videobert_tokens)
        videobert_tokens_mask = tf.ones(dtype=tf.float32, shape=videobert_len)

        return {"video_id": video_id,
                "sentence_id": sen_id,
                "features": images_feats,
                "i3d_features": i3d_feats,
                "aud_features": aud_feats,
                'num_frames_mask': num_frames_mask,
                'num_i3d_frames_mask': num_i3d_frames_mask,
                'num_aud_frames_mask': num_aud_frames_mask,
                "caption_ids": caption_ids,
                "caption_len": caption_len,
                "mask": mask,
                "videobert_tokens": videobert_tokens,
                "videobert_len": videobert_len,
                "videobert_tokens_mask": videobert_tokens_mask,
                "i3d_timestamp_factor": i3d_timestamp_factor,
                "aud_timestamp_factor": aud_timestamp_factor
                }

    dataset = dataset.map(prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    padded_shapes = {
        'video_id': tf.TensorShape([]),
        'sentence_id': tf.TensorShape([]),
        'features': tf.TensorShape(transform_shape(preproc_img['out_shape'])),
        'i3d_features': tf.TensorShape(transform_shape(preproc_i3d['out_shape'])),
        'aud_features': tf.TensorShape(transform_shape(preproc_aud['shape'])),
        'num_frames_mask': tf.TensorShape([None]),
        'num_i3d_frames_mask': tf.TensorShape([None]),
        'num_aud_frames_mask': tf.TensorShape([None]),
        'caption_ids': tf.TensorShape([None]),
        'caption_len': tf.TensorShape([]),
        #'raw_images': tf.TensorShape(transform_shape(preproc_img_raw['out_shape'])),
        'mask': tf.TensorShape([None]),
        'videobert_tokens': tf.TensorShape([None]),
        'videobert_len': tf.TensorShape([]),
        #'raw_images': tf.TensorShape(transform_shape(preproc_img_raw['out_shape'])),
        'videobert_tokens_mask': tf.TensorShape([None]),
        'i3d_timestamp_factor': tf.TensorShape([]),
        'aud_timestamp_factor': tf.TensorShape([])
    }
    padding_values = {
        'video_id': '',
        'sentence_id': '',
        'features': tf.cast(0, dtype=tf.float32),
        'i3d_features': tf.cast(0, dtype=tf.float32),
        'aud_features': tf.cast(0, dtype=tf.float32),
        'num_frames_mask': tf.cast(0, dtype=tf.float32),
        'num_i3d_frames_mask': tf.cast(0, dtype=tf.float32),
        'num_aud_frames_mask': tf.cast(0, dtype=tf.float32),
        'caption_ids': tf.cast(0, dtype=tf.int32),
        'caption_len': tf.cast(-1, dtype=tf.int32),
        'mask': tf.cast(0, dtype=tf.float32),
        'videobert_tokens': tf.cast(0, dtype=tf.int32),
        'videobert_len': tf.cast(-1, dtype=tf.int32),
        'videobert_tokens_mask': tf.cast(0, dtype=tf.float32),
        'i3d_timestamp_factor': tf.cast(1, dtype=tf.float32), # If no timestamp factor available use 1?
        'aud_timestamp_factor': tf.cast(1, dtype=tf.float32), # If no timestamp factor available use 1?
    }


    if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values,
                                       drop_remainder=False)
    else:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values,
                                       drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, len(json_data)


def get_videobert_pretrain_dataset(dataset_path, mode, params, batch_size=32, shuffle=False):
    if os.path.isdir(dataset_path):
        if mode == tf.estimator.ModeKeys.TRAIN:
            json_path = os.path.join(dataset_path, 'train_samples_videobert_pretrain_only.json')
        else:
            json_path = os.path.join(dataset_path, 'val_samples_videobert_pretrain_only.json')

    print("Loading dataset json from disk...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    def my_generator_vbert(json_data):

        video_ids = list(map(lambda x: my_transform(x, 'video_id'), json_data))
        videobert_tokens = list(map(lambda x: my_transform(x, 'videobert_tokens'), json_data))
        for idx in range(len(json_data)):
            yield (video_ids[idx],
                   videobert_tokens[idx])


    generator = lambda: my_generator_vbert(json_data=json_data)


    output_types = (tf.string, tf.int32)
    output_shapes = (
        tf.TensorShape([]),
        tf.TensorShape([None]),
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types, output_shapes=output_shapes)


    if mode == tf.estimator.ModeKeys.TRAIN or shuffle:
        dataset = dataset.shuffle(buffer_size=len(json_data))

    def prepare_sample(video_id,
                       videobert_tokens):


        videobert_len = tf.size(videobert_tokens)
        videobert_tokens_mask = tf.ones(dtype=tf.float32, shape=videobert_len)

        return {"video_id": video_id,
                "sentence_id": "",
                "features": 0.0,
                "i3d_features": 0.0,
                "aud_features": 0.0,
                'num_frames_mask': [0.0],
                'num_i3d_frames_mask': [0.0],
                'num_aud_frames_mask': [0.0],
                "caption_ids": [0],
                "caption_len": 0,
                "mask": [0.0],
                "videobert_tokens": videobert_tokens,
                "videobert_len": videobert_len,
                "videobert_tokens_mask": videobert_tokens_mask
                }

    dataset = dataset.map(prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    padded_shapes = {
        'video_id': tf.TensorShape([]),
        'sentence_id': tf.TensorShape([]),
        'features': tf.TensorShape([]),
        'i3d_features': tf.TensorShape([]),
        'aud_features': tf.TensorShape([]),
        'num_frames_mask': tf.TensorShape([None]),
        'num_i3d_frames_mask': tf.TensorShape([None]),
        'num_aud_frames_mask': tf.TensorShape([None]),
        'caption_ids': tf.TensorShape([None]),
        'caption_len': tf.TensorShape([]),
        'mask': tf.TensorShape([None]),
        'videobert_tokens': tf.TensorShape([None]),
        'videobert_len': tf.TensorShape([]),
        'videobert_tokens_mask': tf.TensorShape([None]),
    }
    padding_values = {
        'video_id': '',
        'sentence_id': '',
        'features': tf.cast(0, dtype=tf.float32),
        'i3d_features': tf.cast(0, dtype=tf.float32),
        'aud_features': tf.cast(0, dtype=tf.float32),
        'num_frames_mask': tf.cast(0, dtype=tf.float32),
        'num_i3d_frames_mask': tf.cast(0, dtype=tf.float32),
        'num_aud_frames_mask': tf.cast(0, dtype=tf.float32),
        'caption_ids': tf.cast(0, dtype=tf.int32),
        'caption_len': tf.cast(-1, dtype=tf.int32),
        'mask': tf.cast(0, dtype=tf.float32),
        'videobert_tokens': tf.cast(0, dtype=tf.int32),
        'videobert_len': tf.cast(-1, dtype=tf.int32),
        'videobert_tokens_mask': tf.cast(0, dtype=tf.float32),
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values,
                                       drop_remainder=False)
    else:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values,
                                       drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, len(json_data)
