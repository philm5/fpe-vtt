# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

We edited the demo file in order to extract features for our datasets.
"""

from __future__ import print_function

import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import glob
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(_):
  ds_base_path = '/path/to/Dataset/base/path/VATEX'

  feat_path = os.path.join(ds_base_path, 'features_audio')
  if not os.path.exists(feat_path):
      os.makedirs(feat_path, exist_ok=True)
  audio_path = os.path.join(ds_base_path, 'audio')
  audio_files = glob.glob(os.path.join(audio_path, '*.wav'))


  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  # If needed, prepare a record writer to store the postprocessed embeddings.
  writer = tf.python_io.TFRecordWriter(
      FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    pbar = tqdm(audio_files)
    for file in pbar:
        examples_batch = vggish_input.wavfile_to_examples(file)
        if examples_batch is None:
            continue
        pbar.set_description(str(examples_batch.shape))

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        #print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        #print(postprocessed_batch, "shape", postprocessed_batch.shape)

        basename = os.path.splitext(os.path.basename(file))[0]

        features_vggish_raw = os.path.join(feat_path, f"{basename}_vggish_raw.bin")
        features_vggish_pca = os.path.join(feat_path, f"{basename}_vggish_pca.bin")

        postprocessed_batch.tofile(features_vggish_pca)
        embedding_batch.tofile(features_vggish_raw)


    #  # Write the postprocessed embeddings as a SequenceExample, in a similar
    # # format as the features released in AudioSet. Each row of the batch of
    # # embeddings corresponds to roughly a second of audio (96 10ms frames), and
    # # the rows are written as a sequence of bytes-valued features, where each
    # # feature value contains the 128 bytes of the whitened quantized embedding.
    # seq_example = tf.train.SequenceExample(
    #     feature_lists=tf.train.FeatureLists(
    #         feature_list={
    #             vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
    #                 tf.train.FeatureList(
    #                     feature=[
    #                         tf.train.Feature(
    #                             bytes_list=tf.train.BytesList(
    #                                 value=[embedding.tobytes()]))
    #                         for embedding in postprocessed_batch
    #                     ]
    #                 )
    #         }
    #     )
    # )
    # print(seq_example)
    # if writer:
    #   writer.write(seq_example.SerializeToString())

  if writer:
    writer.close()

if __name__ == '__main__':
  tf.app.run()
