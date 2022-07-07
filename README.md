# Synchronized Audio-Visual Frames with Fractional Positional Encoding for Transformers in Video-to-Text Translation

Code repository to train our VTT models. This is research code for the ICIP submission mentioned above.

## Prepare Data

Look at the scripts in folder dataset_tools/ :
 * `dataset_tools/01_convert_DS_to_intermediate.py`: converts different datasets to an intermediate format to work with


In order for the following scripts to work, you have to download the datasets yourself. 01_convert_DS_to_intermediate.py creates bash scripts for extracting frames and audio from videos. Before proceeding, you have to extract features with:
 * **I3D**: `kinteics-i3d/extract_features_with_i3d.py`
 * **ResNet**: `extract_features.py`
 * **VGGish**: `vggish/vggish_inference_bin.py`

Then continue with following scripts:
 * `02_create_full_ds_json.py`: Converts different datasets to a json format, which our models can work with in this pipeline
 * `03_create_missing_audio_features.py`: Creates dummy audio features for missing audio clips.
 * `04_create_missing_resnet_features.py`: Creates dummy ResNet features for missing videos.

Then you need to compress ResNet features with: `compress_feature_files_per_video.py`

To use SCST, you need to preprocess the dataset into ngrams with dataset_tools/prepro_ngrams.py. If you want to use pretrained BERT embedding you need to download them and convert them to TensorFlow weights. The BERT vocab comes with these weights. See https://tfhub.dev/google/small_bert/bert_uncased_L-8_H-512_A-8/1 for these weights.

## Train

A training can be started with `train_transformer.py`. Adapt the parameters of the params-dictionary to match your needs. Also adapt the dataset path in Line 23 to a dataset created by script `02_create_full_ds_json.py`.

You can continue training with SCST with the script train_transformer_rl.py. Parameters should be the same and you need to specify a best model checkpoint for the params key:

    finetune': {
        "init_from": "./checkpoints/train/MODEL_NAME/best_model/ckpt-20817",
        'train_ds_json': None,
    },

## Inference

You can predict captions for unknown video datasets with the script `inference_testset.py`