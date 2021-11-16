import json
import os
import re
from tqdm import tqdm
import tensorflow as tf
import random
import datetime
import copy
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import tensorflow_datasets as tfds
from official.nlp.bert import tokenization
import math

dataset_info_fname = 'dataset_info.json'
na_videos_fname = 'unavailable_videos.json'
all_samples_fname = 'all_samples.json'
train_samples_fname = 'train_samples.json'
val_samples_fname = 'val_samples.json'
test_samples_fname = 'test_samples.json'
val_gt_fname = 'val_gt.json'
test_gt_fname = 'test_gt.json'
vocab_fname = 'vocab.json'
subword_vocab_fname = 'subword_vocab'
train_video_ids_fname = 'train_video_ids.json'
val_video_ids_fname = 'train_video_ids.json'
test_video_ids_fname = 'test_video_ids_fname.json'

def load_generic_dataset(base_path, datasplit_name, dst_path, debug=False):
    full_path = os.path.join(base_path, dataset_info_fname)
    with open(full_path, 'r') as f:
        videodatainfo = json.load(f)

    sentences = videodatainfo['sentences']
    videos = videodatainfo['videos']

    all_samples = []
    all_captions = []
    unavailable_videos = []
    if debug:
        videos = videos[:100]

    for video in tqdm(videos):
        video_id = video['video_id']
        if os.path.exists(video['raw_video_path']) and os.stat(video['raw_video_path']).st_size > 0:
            video_infos = {}
            for k, v in video.items():
                video_infos[f'video_infos_{k}'] = v

            video_frames_path = os.path.join(base_path, 'frames', video_id)
            num_frames = len(glob.glob(os.path.join(video_frames_path, '*.jpg')))
            # only use video if it actually exists. If file was N/A the filesize of wget == 0
            per_video_sentences = [x for x in sentences if x['video_id'] == video_id]
            for video_sentence in per_video_sentences:
                # old! :caption = '<start> ' + video_sentence['caption'] + ' <end>'
                caption = video_sentence['caption']
                sample = copy.deepcopy(video_sentence)
                sample['video_id'] = video_id
                sample['caption'] = caption
                sample['base_path'] = base_path
                sample['num_frames'] = num_frames
                # add video information...
                sample.update(video_infos)

                all_captions.append(caption)
                all_samples.append(sample)


        else:
            unavailable_videos.append(video['raw_video_path'])

    na_videos = os.path.join(dst_path, f"{datasplit_name}_{na_videos_fname}")

    print(f"{len(unavailable_videos)} were not available. Dumping them to {na_videos}...")
    with open(na_videos, 'w') as f:
        json.dump(unavailable_videos, f, indent=4)

    return all_samples, all_captions, na_videos

def create_train_val_split(base_path, dst_path, datasplit_name, all_samples, train_portion=0.9, val_portion=0.0, random_seed=7747):
    all_samples_path = os.path.join(dst_path, f"{datasplit_name}_{all_samples_fname}")
    with open(all_samples_path, 'w') as f:
        json.dump(all_samples, f, indent=4)

    if val_portion == 0.0:
        val_portion = 1.0 - train_portion
        test_portion = 0.0
    else:
        assert(val_portion+train_portion <= 1.0)
        test_portion = 1.0 - (train_portion + val_portion)



    # find unique video ids
    samples_per_video_dict = {}
    for s in tqdm(all_samples, desc="Grouping samples based on videos"):
        video_id = s['video_id']
        if video_id not in samples_per_video_dict:
            samples_per_video_dict[video_id] = []
        samples_per_video_dict[video_id].append(s)

    samples_per_video_list = np.asarray([(k, v) for k, v in samples_per_video_dict.items()])
    np.random.seed(random_seed)
    np.random.shuffle(samples_per_video_list)
    split_indices = [int(train_portion * len(samples_per_video_dict)), int((train_portion + val_portion) * len(samples_per_video_dict))]
    train_dict, val_dict, test_dict = np.split(samples_per_video_list, split_indices)

    train_video_ids = [x for x, _ in train_dict]
    val_video_ids = [x for x, _ in val_dict]
    test_video_ids = [x for x, _ in test_dict]

    assert len(np.intersect1d(train_video_ids, val_video_ids)) == 0
    assert len(np.intersect1d(val_video_ids, test_video_ids)) == 0
    assert set(np.union1d(np.union1d(train_video_ids, val_video_ids), test_video_ids)) == set([x for x, _ in samples_per_video_list])

    # dump train and val video ids
    with open(os.path.join(base_path, train_video_ids_fname), 'w') as f:
        json.dump(train_video_ids, f, indent=4)
    with open(os.path.join(base_path, val_video_ids_fname), 'w') as f:
        json.dump(val_video_ids, f, indent=4)
    with open(os.path.join(base_path, test_video_ids_fname), 'w') as f:
        json.dump(test_video_ids, f, indent=4)

    # split actual samples according to the video ids
    train_df = []
    for _, y in train_dict:
        train_df.extend(y)
    val_df = []
    for _, y in val_dict:
        val_df.extend(y)
    test_df = []
    for _, y in test_dict:
        test_df.extend(y)
    # assert if split has correct number of samples
    assert (len(train_df) + len(val_df) + len(test_df) == len(all_samples))

    # now dump the splitted data to disk
    with open(os.path.join(dst_path, f"{datasplit_name}_{train_samples_fname}"), 'w') as f:
        json.dump(train_df, f, indent=4)

    with open(os.path.join(dst_path, f"{datasplit_name}_{val_samples_fname}"), 'w') as f:
        json.dump(val_df, f, indent=4)

    with open(os.path.join(dst_path, f"{datasplit_name}_{test_samples_fname}"), 'w') as f:
        json.dump(test_df, f, indent=4)

    return train_df, val_df, test_df

def create_val_gt(dst_path, datasplit, optional_val_samples_fname=None, optional_val_gt_name=None):
    if optional_val_samples_fname is None:
        optional_val_samples_fname = val_samples_fname

    if optional_val_gt_name is None:
        optional_val_gt_name = val_gt_fname

    json_path = os.path.join(dst_path, optional_val_samples_fname)
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    annots = []
    images = []

    for sample in json_data:
        cap = sample['caption']
        cap = cap.replace("<start> ", "").replace(" <end>", "")
        gt = {"image_id": sample["video_id"], "caption": cap, "id": sample['sen_id']}
        annots.append(gt)
        images.append({"id": sample["video_id"]})

    val_gt = {"annotations": annots, "images": images, "type": "captions", "licenses": [],
              "info":
                  {
                      "description": f"This is a custom VTT dataset val file for following datasplit: {datasplit}",
                      "url": "https://www.uni-augsburg.de/de/fakultaet/fai/informatik/prof/mmc/",
                      "version": "1.0",
                      "year": datetime.datetime.now().year,
                      "contributor": "Uni Augsburg",
                      "date_created": str(datetime.datetime.now())
                  }
              }

    with open(os.path.join(dst_path, optional_val_gt_name), 'w') as f:
        json.dump(val_gt, f, indent=4)

def gen_vocab(vocab_len, all_captions, dst_path, generate_subword_vocab='False'):
    """

    :param vocab_len:
    :param all_captions:
    :param dst_path:
    :param generate_subword_vocab: Can be 'False', 'True' or 'BERT'
    :return:
    """
    print("Generating vocabulary...")
    # Choose the top 5000 words from the vocabulary

    if generate_subword_vocab == 'True':
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_captions, target_vocab_size=vocab_len)

        vocab_file_path = os.path.join(dst_path, subword_vocab_fname)
        print(f"Dumping vocab to {vocab_file_path}...")
        tokenizer_en.save_to_file(vocab_file_path)
        return tokenizer_en
    elif generate_subword_vocab == 'False':
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_len,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(all_captions)
        #train_seqs = tokenizer.texts_to_sequences(captions)
        vocab_file_path = os.path.join(dst_path, vocab_fname)
        print(f"Dumping vocab to {vocab_file_path}...")
        with open(vocab_file_path, 'w') as f:
            dump = {'index_word': tokenizer.index_word,
                    'word_index': tokenizer.word_index}
            json.dump(dump, f, indent=4)

        return tokenizer
    elif generate_subword_vocab == 'BERT':
        model_path_base_dir = 'models/BERT/small_bert_bert_en_uncased_L-8_H-512_A-8_1/'
        vocab_path = os.path.join(model_path_base_dir, 'assets', 'vocab.txt')

        tokenizer = tokenization.FullTokenizer(vocab_path)

        return tokenizer

def add_caption_ids(my_samples, tokenizer, fname, dst_path, sent_id_cnter=1, generate_subword_vocab='False'):
    """

    :param my_samples:
    :param tokenizer:
    :param fname:
    :param dst_path:
    :param sent_id_cnter:
    :param generate_subword_vocab: Can be 'False', 'True' or 'BERT'
    :return:
    """
    #

    print(f"Transforming captions [{fname}] into ids...")
    for sample in tqdm(my_samples):
        caption = sample['caption']

        if generate_subword_vocab == 'True':
            start_id = tokenizer.vocab_size
            end_id = tokenizer.vocab_size + 1
            caption_ids = [start_id] + tokenizer.encode(caption) + [end_id]
        elif generate_subword_vocab == 'False':
            start_id = tokenizer.num_words
            end_id = tokenizer.num_words + 1
            caption_ids = [start_id] + tokenizer.texts_to_sequences([caption])[0] + [end_id]
        elif generate_subword_vocab == 'BERT':
            start_id = tokenizer.convert_tokens_to_ids(['[unused0]'])[0]
            end_id = tokenizer.convert_tokens_to_ids(['[unused1]'])[0]
            if type(caption) == float:
                print()
            tokens = tokenizer.tokenize(caption)
            caption_ids = [start_id] + tokenizer.convert_tokens_to_ids(tokens) + [end_id]

        sample['caption_ids'] = caption_ids
        sample['sen_id'] = f"{sample['video_id']}_{sent_id_cnter}"
        sent_id_cnter += 1

    all_samples_path = os.path.join(dst_path, fname)
    with open(all_samples_path, 'w') as f:
        json.dump(my_samples, f, indent=4)

    return sent_id_cnter


def load_single_ds(base_path, datasplit, dst_path, train_portion=0.9, val_portion=0.0, debug=False):
    all_samples, all_captions, _ = load_generic_dataset(base_path=base_path, datasplit_name=datasplit,
                                                                        dst_path=dst_path, debug=debug)
    if debug:
        all_samples = all_samples[:100]

    train_split, val_split, test_split = create_train_val_split(base_path=base_path,
                                                      dst_path=dst_path,
                                                      datasplit_name=datasplit,
                                                      all_samples=all_samples,
                                                      train_portion=train_portion,
                                                    val_portion=val_portion)

    return all_samples, all_captions, train_split, val_split, test_split


def create_final_train_val_dataset(datasplit_config, dst_path, new_datasplit_name, vocab_len=10000, debug=False, generate_subword_vocab="False"):
    """

    :param datasplit_config:
    :param dst_path:
    :param new_datasplit_name:
    :param vocab_len:
    :param debug:
    :param generate_subword_vocab: Can be 'False', 'True' or 'BERT'
    :return:
    """

    train_all_df = []
    val_all_df = []
    test_all_df = []
    all_captions = []
    prepared_datasplits = {}

    for ds in datasplit_config:
        ds_all_samples, ds_captions, train_split, val_split, test_split = load_single_ds(base_path=ds['base_path'],
                                                                             datasplit=ds['datasplit'],
                                                                             train_portion=ds['train_portion'],
                                                                             val_portion=ds['val_portion'],
                                                                              dst_path=dst_path,
                                                                             debug=debug)

        if 'ds_multiplier' in ds:
            train_split = list(np.repeat(train_split, ds['ds_multiplier']))
            val_split = list(np.repeat(val_split, ds['ds_multiplier']))
            test_split = list(np.repeat(test_split, ds['ds_multiplier']))

        prepared_datasplits[ds['datasplit']] = (train_split, val_split, test_split)


        train_all_df.extend(train_split)
        val_all_df.extend(val_split)
        test_all_df.extend(test_split)
        all_captions.extend(ds_captions)

    tokenizer = gen_vocab(vocab_len=vocab_len,
                          all_captions=all_captions,
                          dst_path=dst_path,
                          generate_subword_vocab=generate_subword_vocab)

    # Create extra files for each datasplit
    for datasplit_name, splits in prepared_datasplits.items():
        train_split, val_split, test_split = splits

        datasplit_subdir_name = os.path.join(dst_path, datasplit_name)
        os.makedirs(datasplit_subdir_name, exist_ok=True)

        train_samples_subdir_fname = os.path.join(datasplit_name, train_samples_fname)
        val_samples_subdir_fname = os.path.join(datasplit_name, val_samples_fname)
        test_samples_subdir_fname = os.path.join(datasplit_name, test_samples_fname)
        val_gt_subdir_fname = os.path.join(datasplit_name, val_gt_fname)
        test_gt_subdir_fname = os.path.join(datasplit_name, test_gt_fname)

        sent_id_cnter = add_caption_ids(train_split,
                                        tokenizer,
                                        fname=train_samples_subdir_fname,
                                        dst_path=dst_path,
                                        sent_id_cnter=1,
                                        generate_subword_vocab=generate_subword_vocab)
        _ = add_caption_ids(val_split,
                            tokenizer,
                            fname=val_samples_subdir_fname,
                            dst_path=dst_path,
                            sent_id_cnter=sent_id_cnter,
                            generate_subword_vocab=generate_subword_vocab)

        _ = add_caption_ids(test_split,
                            tokenizer,
                            fname=test_samples_subdir_fname,
                            dst_path=dst_path,
                            sent_id_cnter=sent_id_cnter,
                            generate_subword_vocab=generate_subword_vocab)

        create_val_gt(dst_path=dst_path,
                      datasplit=datasplit_name,
                      optional_val_samples_fname=val_samples_subdir_fname,
                      optional_val_gt_name=val_gt_subdir_fname)

        create_val_gt(dst_path=dst_path,
                      datasplit=datasplit_name,
                      optional_val_samples_fname=test_samples_subdir_fname,
                      optional_val_gt_name=test_gt_subdir_fname)

    sent_id_cnter = add_caption_ids(train_all_df,
                                    tokenizer,
                                    fname=train_samples_fname,
                                    dst_path=dst_path,
                                    sent_id_cnter=1,
                                    generate_subword_vocab=generate_subword_vocab)
    _ = add_caption_ids(val_all_df,
                        tokenizer,
                        fname=val_samples_fname,
                        dst_path=dst_path,
                        sent_id_cnter=sent_id_cnter,
                        generate_subword_vocab=generate_subword_vocab)
    _ = add_caption_ids(test_all_df,
                        tokenizer,
                        fname=test_samples_fname,
                        dst_path=dst_path,
                        sent_id_cnter=sent_id_cnter,
                        generate_subword_vocab=generate_subword_vocab)
    create_val_gt(dst_path=dst_path,
                  datasplit=new_datasplit_name)
    create_val_gt(dst_path=dst_path,
                  datasplit=new_datasplit_name,
                  optional_val_samples_fname=test_samples_fname,
                  optional_val_gt_name=test_gt_fname)


def create_MSRVTT_100percent_trecvid_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/MSR_VTT_full_0.9TRECVID_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="MSR_VTT_full_0.9TRECVID_BERT",
                                   generate_subword_vocab="BERT")


def create_MSR_VTT_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/MSR_VTT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=10000,
                                   new_datasplit_name="MSR_VTT")

def create_trecvid_test():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020_Test',
            'datasplit': 'TRECVID_VTT_2020_Test',
            'train_portion': 0.0
          },

    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2020_Test'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=10000,
                                   new_datasplit_name="TRECVID_VTT_2020_Test")

def create_MSRVTT_100_ACM100_trecvid_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/ACM_GIF',
            'datasplit': 'ACM_GIF',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/MSR_VTT_full_ACM_GIF_full_0.9TRECVID'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=20000,
                                   new_datasplit_name="MSR_VTT_full_ACM_GIF_full_0.9TRECVID",
                                   debug=False,
                                   generate_subword_vocab='True')

def create_vatex():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 0.0
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-official-split_subword_vocab'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-official-split_subword_vocab",
                                   generate_subword_vocab="True") #BERT

def create_VATEX_official_split_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0,
            'val_portion': 0.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 0.0,
            'val_portion': 1.0
          },
    ]


    dst_path = '/path/to/Datasets/VATEX-official-split-v3_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-official-split-v2_BERT",
                                   generate_subword_vocab="BERT") #BERT

def create_vatex_trval_publictest():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0,
            'val_portion': 0.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 1.0,
            'val_portion': 0.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_test',
            'datasplit': 'VATEX_test',
            'train_portion': 0.0,
            'val_portion': 1.0
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-trval-publictest-mysplit-v3_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-trval-publictest-mysplit_BERT",
                                   generate_subword_vocab="BERT") #BERT

def create_vatex_trpt_val():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0,
            'val_portion': 0.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_test',
            'datasplit': 'VATEX_test',
            'train_portion': 1.0,
            'val_portion': 0.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 0.0,
            'val_portion': 1.0
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-trpt_val_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-trpt_val_BERT",
                                   generate_subword_vocab="BERT") #BERT




def create_vatex_public_test():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_test',
            'datasplit': 'VATEX_test',
            'train_portion': 0.0,
            'val_portion': 1.0
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-official-split_BERT_public_test-v2'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-official-split_BERT_public_test-v2",
                                   generate_subword_vocab="BERT") #BERT

def create_vatex_private_test():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_private_test',
            'datasplit': 'VATEX_private_test',
            'train_portion': 0.0,
            'val_portion': 1.0
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-official-split_BERT_private_test-v2'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-official-split_BERT_private_test-v2",
                                   generate_subword_vocab="BERT") #BERT


def create_vatex_train_trecvid_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-full-TRECVID0.9_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-full-TRECVID0.9_BERT",
                                   generate_subword_vocab="BERT")

def create_vatex_msr_trecvid_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-full-MSR-full-TRECVID0.9_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-full-MSR-full-TRECVID0.9_BERT",
                                   generate_subword_vocab="BERT")


def create_vatex_msr_acm_trecvid_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/VATEX_val',
            'datasplit': 'VATEX_val',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/ACM_GIF',
            'datasplit': 'ACM_GIF',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/VATEX-full-MSR-full-ACM-gif-TRECVID0.9_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="VATEX-full-MSR-full-ACM-gif-TRECVID0.9_BERT",
                                   generate_subword_vocab="BERT")

def create_MSR_VTT_val_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 0.9,
            'val_portion': 0.1
          },

    ]


    dst_path = '/path/to/Datasets/MSR_VTT_0.9_val-v3_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="MSR_VTT_0.9_val-v3_BERT",
                                   generate_subword_vocab="BERT")

def create_MSR_VTT_fullval_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 0.0
          },

    ]


    dst_path = '/path/to/Datasets/MSR_VTT_fullval_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="MSR_VTT_0.9_val_BERT",
                                   generate_subword_vocab="BERT")


def create_TRECVID_val_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2020_0.9_val_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2020_0.9_val_BERT",
                                   generate_subword_vocab="BERT")

def create_TRECVID_fullval_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2020',
            'datasplit': 'TRECVID_VTT_2020',
            'train_portion': 0.0
          },

    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2020_fullval_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2020_fullval_BERT",
                                   generate_subword_vocab="BERT")


def create_TRECVID2021_BERT_90percent():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021',
            'datasplit': 'TRECVID_VTT_2021',
            'train_portion': 0.9
          },

    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2021_BERT_90percent'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2021_BERT_90percent",
                                   generate_subword_vocab="BERT")


def create_TRECVID_VTT_2021_90percent_VATEX_train_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
        },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021',
            'datasplit': 'TRECVID_VTT_2021',
            'train_portion': 0.9
          },
    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2021_90percent+VATEX_train_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2021_90percent+VATEX_train_BERT",
                                   generate_subword_vocab="BERT")


def create_5xTRECVID_VTT_2021_90percent_VATEX_train_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
        },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021',
            'datasplit': 'TRECVID_VTT_2021',
            'train_portion': 0.9,
            'ds_multiplier': 5
          },
    ]


    dst_path = '/path/to/Datasets/5xTRECVID_VTT_2021_90percent_VATEX_train_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="5xTRECVID_VTT_2021_90percent_VATEX_train_BERT",
                                   generate_subword_vocab="BERT")


def create_5xTRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
        },
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021',
            'datasplit': 'TRECVID_VTT_2021',
            'train_portion': 0.9,
            'ds_multiplier': 5
          },
    ]


    dst_path = '/path/to/Datasets/5xTRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="5xTRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT",
                                   generate_subword_vocab="BERT")


def create_TRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/VATEX_train',
            'datasplit': 'VATEX_train',
            'train_portion': 1.0
        },
        {
            'base_path': '/path/to/Datasets/MSR_VTT',
            'datasplit': 'MSR_VTT',
            'train_portion': 1.0
          },
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021',
            'datasplit': 'TRECVID_VTT_2021',
            'train_portion': 0.9
          },
    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT",
                                   generate_subword_vocab="BERT")


def create_TRECVID_VTT_2021_test_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/TRECVID_VTT_2021_Test',
            'datasplit': 'VATEX_train',
            'train_portion': 0.0
        },
    ]


    dst_path = '/path/to/Datasets/TRECVID_VTT_2021_test_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="TRECVID_VTT_2021_test_BERT",
                                   generate_subword_vocab="BERT")

def create_MSVD_val_BERT():
    datasplits = [
        {
            'base_path': '/path/to/Datasets/MSVDD',
            'datasplit': 'MSVD',
            'train_portion': 1200.0/1970.0,
            'val_portion': 100.0/1970.0,
          },

    ]


    dst_path = '/path/to/Datasets/MSVD_official-split_100val_BERT'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    create_final_train_val_dataset(datasplit_config=datasplits,
                                   dst_path=dst_path,
                                   vocab_len=12000,
                                   new_datasplit_name="MSVD_official-split_100val_BERT",
                                   generate_subword_vocab="BERT")
#create_MSRVTT_100percent_trecvid_90percent()
#create_MSRVTT_100percent_trecvid_90percent()
#create_MSRVTT_100_ACM100_trecvid_90percent()

#create_trecvid_test()

#create_vatex_train_trecvid_90percent()

#create_vatex_msr_trecvid_90percent()

#create_vatex_msr_acm_trecvid_90percent()

#create_vatex()

#create_vatex_trval_publictest()

#create_vatex_public_test()

#create_vatex_private_test()

#create_TRECVID_val_BERT()
#create_TRECVID_fullval_BERT()

#create_MSR_VTT_fullval_BERT()

#create_MSR_VTT_val_BERT()

#create_TRECVID2021_BERT_90percent()

#create_TRECVID_VTT_2021_90percent_VATEX_train_BERT()

#create_5xTRECVID_VTT_2021_90percent_VATEX_train_BERT()

#create_5xTRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT()
#create_TRECVID_VTT_2021_90percent_VATEX_MSR_VTT_train_BERT()

#create_TRECVID_VTT_2021_test_BERT()

#create_MSVD_val_BERT()

#create_vatex_trval_publictest()
#create_VATEX_official_split_BERT()

#create_MSVD_val_BERT()

create_vatex_trpt_val()