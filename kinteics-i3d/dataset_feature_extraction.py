import tensorflow.compat.v1 as tf
import os
import json
from tqdm import tqdm
import re
import glob
from itertools import islice
import natsort


"""
Extract I3D features for our dataset format. I.e., generates a dataset object compatible with our DS.
"""

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

# new generic ds format
def get_dataset_feat_ext(ds_base_path, video_segment_len=None, fps=None):
    if video_segment_len is not None and fps is not None:
        #video_segment_len = 1.5
        #fps = 20
        num_frames_per_split = int(video_segment_len * fps)
    else:
        num_frames_per_split = None

    json_ds_info = os.path.join(ds_base_path, 'dataset_info.json')
    with open(json_ds_info, 'r') as f:
        ds_info = json.load(f)

    videos = ds_info['videos']
    videos_with_frames = {}
    for vid in videos:
        vid_frames_path = os.path.join(ds_base_path, 'frames', vid['video_id'])
        vid_frames_jpg_paths = glob.glob(os.path.join(vid_frames_path, '*.jpg'))
        if 0 < len(vid_frames_jpg_paths) < 16:
            videos_with_frames[vid['video_id']] = [vid_frames_jpg_paths[0]] * 16
        elif len(vid_frames_jpg_paths) > 0:
            videos_with_frames[vid['video_id']] = vid_frames_jpg_paths
        else:
            print(vid_frames_path)

    regex_frame_no = re.compile("(\d+)\.jpg")

    vnames = []
    framess = []

    ntkeygen = natsort.natsort_keygen(key=lambda y: y.lower())
    for video, frames in tqdm(videos_with_frames.items()):
        frames.sort(key=ntkeygen)
        vnames.append(video)
        if len(frames) > 1000:
            framess.append(frames[0:1000])
        else:
            framess.append(frames)

    def my_generator_splits(vnames, framess):
        for idx in range(len(vnames)):
            if len(framess[idx]) > 0:
                fpaths = framess[idx]
                for i in range(0, len(fpaths), num_frames_per_split):
                    yield (vnames[idx], i, fpaths[i:i + num_frames_per_split])

    def my_generator(vnames, framess):
        for idx in range(len(vnames)):
            if len(framess[idx]) > 0:
                yield (vnames[idx], -1, framess[idx])


    #vnames_altered = [x[0] for x in zip(vnames, framess) if len(x[1]) > 0]
    #vnames_altered = vnames

    output_types = (tf.string, tf.int32, tf.string)
    output_shapes = (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None]))


    if num_frames_per_split is not None:
        gen = my_generator_splits
    else:
        gen = my_generator

    ds = tf.data.Dataset.from_generator(
        lambda: gen(vnames, framess),
        output_types=output_types, output_shapes=output_shapes)

    ds = ds.map(map_func=lambda vid, split, fnames: read_seq(vid, split, fnames), num_parallel_calls=16)
    # dataset = dataset.map(map_func=lambda item: (item, tf.py_func(read_npy_file, [item], [tf.float32])), num_parallel_calls=64)
    #ds = ds.map(lambda x: read_image(x, tf.estimator.ModeKeys.EVAL))
    #ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)

    res = list(gen(vnames, framess))

    return ds, len(res)

def resize_image_keep_aspect(image, lo_dim=256):
    # Take width/height
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    # Take the greater value, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.cast(min_, dtype=tf.float32) / tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.cast(tf.cast(initial_width, dtype=tf.float32) / ratio, dtype=tf.int32)
    new_height = tf.cast(tf.cast(initial_height, dtype=tf.float32) / ratio, dtype=tf.int32)

    return tf.image.resize_images(image, [new_width, new_height])

def crop_center(image, crop_size=224):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    y_begin = (h - crop_size) // 2
    x_begin = (w - crop_size) // 2

    crop = tf.image.crop_to_bounding_box(image, y_begin, x_begin, crop_size, crop_size)
    return crop

def read_seq(vname, split, fnames):
    images = tf.map_fn(read_image, fnames, dtype=tf.float32)
    images = tf.stack(images)
    images = tf.expand_dims(images, axis=0)
    #fname = tf.Print(fname, [fname, vname], summarize=100, message="READIMG")
    return vname, split, images

def read_image(fname):
    image_raw = tf.io.read_file(fname)
    image_decoded = tf.image.decode_jpeg(image_raw, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     image_decoded = tf.image.resize(image_decoded, [256, 256])
    #     image_decoded = tf.image.random_crop(image_decoded, [224, 224, 3])
    #     image_decoded = tf.image.random_flip_left_right(image_decoded)
    # else:
    image_decoded = resize_image_keep_aspect(image_decoded)
    image_decoded = crop_center(image_decoded)

    #image_decoded = tf.image.resize(image_decoded, [224, 224])

    image_decoded = tf.subtract(image_decoded, 0.5)
    image_decoded = tf.multiply(image_decoded, 2.0)
    return image_decoded