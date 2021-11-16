import nltk as nltk
import re
import tensorflow as tf
import tensorflow_probability as tfp
from vtt_transformer import ImageCaptioningTransformer, ImageCaptioningTransformerOfficial, ImageEncoder
from simple_model import SimpleModel
from reward import get_self_critical_reward, init_scorer
import os
import json
from calc_scores import calculate_scores, init_coco
from tensorboard.plugins import projector
import pathlib
import glob
import shutil
import tensorflow_datasets as tfds
from official.nlp.bert import tokenization
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from official.nlp.transformer import metrics
from official.nlp.transformer import optimizer
import numpy as np
import sys

max_iters = 30



def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_padding_mask_frames(nfm):
    batch_size = tf.shape(nfm)[0]
    nfm_transformer = 1. - nfm
    nfm_transformer = tf.reshape(nfm_transformer, (batch_size, 1, 1, -1))
    nfm_transformer = tf.cast(nfm_transformer, dtype=tf.float32)
    return nfm_transformer

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.cast(mask, tf.float32)
    return mask  # (seq_len, seq_len)

def create_masks(num_frames_mask, tar_seq):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask_frames(num_frames_mask)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask_frames(num_frames_mask)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_seq)[1])
    dec_target_padding_mask = create_padding_mask(tar_seq)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def get_word(c, dictionary):
    if c in dictionary:
        return dictionary[c]
    else:
        return "<UNK>"


# Define optimizer and loss function -------------------------------------------------------------------------------
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        min = tf.cast(tf.math.minimum(arg1, arg2), dtype=tf.float32)

        return tf.math.rsqrt(self.d_model) * min

class CustomScheduleWithCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,  first_decay_steps, d_model, t_mul=2.0, m_mul=1.0, alpha=0.0, warmup_steps=4000, ):
        super(CustomScheduleWithCosine, self).__init__()
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = tf.math.rsqrt(self.d_model) * (warmup_steps ** -1.5) * warmup_steps

        self.cosineDecayRestarts = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=self.max_lr, first_decay_steps=first_decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha
        )

    def __call__(self, step):
        step_wo_warmup = tf.cast(tf.math.maximum(step - self.warmup_steps, 0), dtype=tf.float32)

        warmup_phase_schedule = step * (self.warmup_steps ** -1.5) * tf.math.rsqrt(self.d_model)
        warmup_phase_mask = tf.cast(step < self.warmup_steps, dtype=tf.float32)

        cosine_decay_restarts_schedule = self.cosineDecayRestarts(step_wo_warmup)
        cosine_decay_restarts_mask = 1 - warmup_phase_mask
        #return arg1mask * arg1

        lr = warmup_phase_mask * warmup_phase_schedule + cosine_decay_restarts_mask * cosine_decay_restarts_schedule
        return lr

class TransformerTools():
    def __init__(self, params, dist_train_dataset, dist_eval_dataset, strategy, coco, mode, model_type="default", pretrain=False):
        self.max_cider_score = -100.00
        self.max_pretrain_score = -100.00
        self.params = params
        self.mode = mode
        self.coco = coco

        self.dist_train_dataset = dist_train_dataset
        self.dist_eval_dataset = dist_eval_dataset
        self.strategy = strategy

        self.model_type = model_type
        #self.use_official = use_official

        if pretrain:
            self.init_pretrain_model()
        else:
            if 'use_subword_vocab' not in self.params:
                self.params['use_subword_vocab'] = False

            self.init_dictionary()
            self.init_model()

            # do some self critical init
            gt_json = os.path.join(self.params['dataset_path'], 'train_samples.json')
            mygts = json.load(open(gt_json, 'r'))
            imgs = {}
            for gt in mygts:
                vid = gt['video_id']
                if vid not in imgs:
                    imgs[vid] = []

                # get caption ids for every corresponding gt without the leading SOS token
                imgs[vid].append(gt['caption_ids'][1:])

            self.imgs_gt = imgs

            scorer_ngrams_path = os.path.join(self.params['dataset_path'], 'ngrams.pkl-idxs')
            if not os.path.exists(scorer_ngrams_path):
                # use fallback path for n-grams, if we did not create them!
                # Create ngrams.pkl-idx: see prepro_ngrams.py
                ngrams_fallback_path = 'VATEX-official-split_BERT/ngrams.pkl-idxs' 
                print(f'{scorer_ngrams_path} not found! Using fallback path {ngrams_fallback_path}.')
                scorer_ngrams_path = ngrams_fallback_path

            init_scorer(scorer_ngrams_path)

    def init_dictionary(self):

        ds_path = self.params['dataset_path']
        self.params['pad_id'] = 0
        if self.params['use_subword_vocab'] == 'True':
            self.dictionary = {}
            self.reverse_dictionary = {}

            self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(ds_path, 'subword_vocab'))
            self.params['vocab_size'] = self.tokenizer.vocab_size
            self.params['start_id'] = self.params['vocab_size']
            self.params['end_id'] = self.params['vocab_size'] + 1

            for i in range(self.params['vocab_size']):
                self.dictionary[i] = self.tokenizer.decode([i])

            self.dictionary[self.params['start_id']] = '<start>'
            self.dictionary[self.params['end_id']] = '<end>'

            self.reverse_dictionary = {v: int(k) for k, v in self.dictionary.items()}
        elif self.params['use_subword_vocab'] == 'False':
            with open(os.path.join(ds_path, 'vocab.json'), 'r') as f:
                vocab = json.load(f)

            self.dictionary = {int(k): v for k, v in vocab['index_word'].items()}
            self.reverse_dictionary = {k: int(v) for k, v in vocab['word_index'].items()}

            if '<start>' in self.reverse_dictionary:
                self.params['start_id'] = self.reverse_dictionary['<start>']
                self.params['end_id'] = self.reverse_dictionary['<end>']
            else:
                self.params['start_id'] = self.params['vocab_size']
                self.params['end_id'] = self.params['vocab_size'] + 1
        elif self.params['use_subword_vocab'] == 'BERT':
            # TODO: replace path
            model_path_base_dir = 'models/BERT/small_bert_bert_en_uncased_L-8_H-512_A-8_1/'
            vocab_path = os.path.join(model_path_base_dir, 'assets', 'vocab.txt')

            tokenizer = tokenization.FullTokenizer(vocab_path)
            self.params['vocab_size'] = len(tokenizer.vocab)

            self.dictionary = tokenizer.inv_vocab
            self.reverse_dictionary = tokenizer.vocab

            self.params['start_id'] = self.reverse_dictionary['[unused0]']
            self.params['end_id'] = self.reverse_dictionary['[unused1]']

        if self.params['use_subword_vocab'] != 'BERT' and ('<start>' not in self.reverse_dictionary or self.params['use_subword_vocab'] == 'True'):
            # Increment vocab size for both vocabs, i.e., add start and end token
            self.params['vocab_size'] += 2  # Account for start and end token

    def init_model(self):
        with self.strategy.scope():
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                            reduction=tf.keras.losses.Reduction.NONE)
                batch_size = self.params['batch_size']

                def loss_function(real, pred):
                    pred = tf.cast(pred, dtype=tf.float32)
                    per_example_loss = loss_object(y_true=real, y_pred=pred)
                    #tf.print("per_example_loss", per_example_loss, per_example_loss.dtype)
                    mask = tf.math.logical_not(tf.math.equal(real, 0))
                    mask = tf.cast(mask, dtype=per_example_loss.dtype)

                    per_example_loss *= mask
                    per_example_loss = tf.reduce_sum(per_example_loss, axis=1)
                    per_example_num_words = tf.reduce_sum(mask, axis=1)
                    per_example_loss /= per_example_num_words

                    avg_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

                    #tf.print("avg_loss", avg_loss)
                    return avg_loss

                self.loss_function = loss_function

                self.add_metrics()

                self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='train_accuracy')

                # Create model, optimizer and checkpoint manager -------------------------------------------------------------------
                # TODO: Remove LSTM
                if self.params['model_type'] == 'LSTM':
                    batch_size = self.params['batch_size']
                    num_samples_per_epoch = self.params['num_samples_per_epoch']
                    # num_epochs_per_decay = params['num_epochs_per_decay']
                    num_steps_per_epoch = num_samples_per_epoch / batch_size
                    decay_steps = num_steps_per_epoch * self.params['num_epochs_per_decay']

                    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.params['learning_rate'],
                                                                        decay_steps=decay_steps,
                                                                        decay_rate=self.params['decay_rate'],
                                                                        staircase=True)
                else:
                    if self.params['lr_schedule'] == 'default_transformer':
                        lr_schedule = CustomSchedule(self.params['d_model'],
                                                      warmup_steps=self.params['transformer_warmup_steps'])
                    elif self.params['lr_schedule'] == 'cosine_warmup':
                        lr_schedule = CustomScheduleWithCosine(
                            first_decay_steps=self.params['train_steps_per_epoch'] * 5,
                            d_model=self.params['d_model'],
                            t_mul=1.0,
                            m_mul=1.0,
                            alpha=0.0,
                            warmup_steps=self.params['transformer_warmup_steps']
                        )

                if self.params['init_word_embedding'] is not None:
                    train_word_embedding = False  # TODO: Change back to False
                else:
                    train_word_embedding = True

                if 'finetune' in self.params:
                    # TODO: finetune in case of RL
                    if not self.params['finetune'] is None:
                        lr_schedule = 5e-6 # 2e-4
                        train_word_embedding = False
                        #train_word_embedding = True

            else:
                lr_schedule = 0.001 # some dummy value

            self.params['train_word_embedding'] = train_word_embedding
            optimizer_tmp = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #, beta_1=0.9, beta_2=0.997, epsilon=1e-9)

            self.optimizer = optimizer_tmp

            if self.model_type == "official":
                # TODO: not implemented
                pass
            elif self.model_type == "default" or self.model_type == 'videobert' or self.model_type == 'xla':
                self.model = ImageCaptioningTransformer(
                    num_layers=self.params['num_layers'],
                    d_model=self.params['d_model'],
                    num_heads=self.params['num_heads'],
                    dff=self.params['dff'],
                    target_vocab_size=self.params['vocab_size'],
                    pe_input=self.params['pe_input'],  # max num_frames
                    pe_target=self.params['pe_target'],  # max tgt sentence length
                    memory_vector_size=self.params['memory_vector_size'],
                    params=self.params,
                )

    def add_metrics(self):
        self.train_loss = tf.keras.metrics.Mean(name='train/train_loss')
        self.global_norm = tf.keras.metrics.Mean(name='train/global_norm')
        self.avg_reward = tf.keras.metrics.Mean(name='stats/avg_reward')
        self.avg_train_ciderd = tf.keras.metrics.Mean(name='stats/avg_train_cider-D')
        self.train_sample_sentence_length = tf.keras.metrics.Mean(name='stats/train_sample_sentence_length')
        self.greedy_sample_sentence_length = tf.keras.metrics.Mean(name='stats/greedy_sample_sentence_length')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

    def train_metrics(self):
        custom_metrics = [self.train_loss,
                self.global_norm,
                self.avg_reward,
                self.avg_train_ciderd,
                self.train_sample_sentence_length,
                self.greedy_sample_sentence_length,
                self.train_accuracy]
        custom_metrics = [x for x in custom_metrics if x is not None]
        return custom_metrics + self.model.metrics

    def predict(self, input):
        if self.model_type == 'default' or self.model_type == 'xla':
            return self.predict_default(input)
  

    def predict_default(self, inputs):
        if not self.params['use_i3d_instead_of_imgs']:
            image_features = inputs[self.params['image_feature_name']]
            frames_mask = inputs[self.params['frame_mask_feature_name']]
        else:
            image_features = inputs[self.params['i3d_feature_name']]
            frames_mask = inputs[self.params['i3d_frame_mask_feature_name']]

            if self.params['i3d_feats'] == 'features_i3d_spatial_rgb_imagenet':
                # shape [BS, -1, 7, 7, 1024] ==> [BS, -1, 1024]
                image_features = tf.reduce_mean(image_features, axis=[2,3])

        batch_size = tf.shape(image_features)[0]
        input_ids = tf.tile([self.params['start_id']], [batch_size])

        if self.model_type == "official":  # self.use_official:
            output = tf.expand_dims(input_ids, axis=1)
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(frames_mask, output)

            result = self.model(inp=image_features,
                       tar=None,
                       training=False,
                       enc_padding_mask=enc_padding_mask,
                       look_ahead_mask=None,
                       dec_padding_mask=None)

            return result[0]['outputs'], inputs['video_id']

        elif self.model_type == "default" or self.model_type == 'xla':
            # Create masks every it.
            enc_padding_mask, _, _ = create_masks(frames_mask, tf.expand_dims(input_ids, axis=1))


            if self.params['use_aud_feats']:
                audio_features = inputs['aud_features']
                aud_frames_mask = inputs['num_aud_frames_mask']
                aud_enc_padding_mask = create_padding_mask_frames(aud_frames_mask)

                gx, x, mask = self.model.encode_step(inp=image_features,
                                                    training=False,
                                                    enc_padding_mask=enc_padding_mask,
                                                    audio_inp=audio_features,
                                                    audio_mask=aud_enc_padding_mask,
                                                    i3d_timestamp_factor=inputs['i3d_timestamp_factor'],
                                                    aud_timestamp_factor=inputs['aud_timestamp_factor'])
            else:
                gx, x, mask = self.model.encode_step(inp=image_features,
                                                    training=False,
                                                    enc_padding_mask=enc_padding_mask,
                                                    i3d_timestamp_factor=inputs['i3d_timestamp_factor'],
                                                    aud_timestamp_factor=inputs['aud_timestamp_factor'])

            enc_output = (gx, x)
            audio_video_dec_padding_mask = mask
            greedy_res, _ = self.greedy_sample(input_ids=input_ids,
                                               enc_output=enc_output,
                                               frames_mask=frames_mask,
                                               train_mode=False,
                                               sample_method="greedy",
                                               dec_padding_mask=audio_video_dec_padding_mask)
            return greedy_res, inputs['video_id']

    def greedy_sample(self,
                      input_ids,
                      enc_output,
                      frames_mask,
                      dec_padding_mask,
                      sample_method="greedy",
                      train_mode=False):


        output = tf.expand_dims(input_ids, axis=1)
        seq_logprobs = tf.TensorArray(dtype=tf.float32,
                                      size=max_iters)

        batch_size = tf.shape(input_ids)[0]
        #self.model.decoder.init_buffer(batch_size)
        for loop_idx in tf.range(start=0, limit=max_iters, delta=1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(output, tf.TensorShape([None, None]))]
            )

            enc_padding_mask, combined_mask, dec_padding_mask_invalid = create_masks(frames_mask,
                                                                             output)

            logprobs, predicted_id = self.greedy_sample_single_step(combined_mask=combined_mask,
                                                                    dec_padding_mask=dec_padding_mask,
                                                                    enc_output=enc_output,
                                                                    eos_idx=self.params['end_id'],
                                                                    loop_idx=loop_idx,
                                                                    output=output,
                                                                    pad_idx=self.params['pad_id'],
                                                                    sample_method=sample_method,
                                                                    train_mode=train_mode)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
            seq_logprobs = seq_logprobs.write(index=loop_idx, value=logprobs)

        #self.model.decoder.clear_buffer(batch_size)

        return output, seq_logprobs.stack()

    def greedy_sample_single_step(self, combined_mask, dec_padding_mask, enc_output, eos_idx, loop_idx, output, pad_idx,
                                  sample_method, train_mode):
        predictions, _ = self.model.decode_step(enc_output=enc_output,
                                                tar=output,
                                                training=train_mode,
                                                look_ahead_mask=combined_mask,
                                                dec_padding_mask=dec_padding_mask)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        # logprobs
        temperature = 1.0
        logprobs = tf.nn.log_softmax(predictions)
        logprobs = logprobs / temperature
        if sample_method == "greedy":
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        else:
            # converted from ImageCaptioning.pytorch (https://github.com/ruotianluo/ImageCaptioning.pytorch)
            cat = tfp.distributions.Categorical(logits=logprobs)
            predicted_id = cat.sample()

        # stop when all finished
        if loop_idx == 0:
            unfinished = predicted_id != eos_idx
        else:
            unfinished = (output[:, loop_idx] != pad_idx) & (output[:, loop_idx] != eos_idx)
            dbg_pt = tf.reduce_any(~unfinished)
            idcs = tf.where(~unfinished)
            pad_words = tf.fill(dims=tf.shape(idcs), value=pad_idx)
            predicted_id = tf.tensor_scatter_nd_update(predicted_id, idcs, pad_words)
            # predicted_id[~unfinished] = pad_idx
            unfinished = unfinished & (predicted_id != eos_idx)  # changed
        return logprobs, predicted_id

    def reward_criterion(self, input, seq, reward):
        """
            # converted from ImageCaptioning.pytorch (https://github.com/ruotianluo/ImageCaptioning.pytorch)
        :param input: sequence logprobs
        :param seq: greedy_sample_res, i.e., the result of categorical sampling of the 5 captions
        :param reward: reward for each sampled caption / word per sampled sente3nce
        :return:
        """

        input = tf.transpose(tf.squeeze(input), perm=[1, 0, 2])
        input = tf.gather(params=input, indices=seq, axis=2, batch_dims=2) # logprobs for each sampled word
        input = tf.reshape(input, [-1])
        reward = tf.reshape(reward, [-1])

        mask = tf.cast(seq > 0, dtype=tf.float32)  # if word > 0 ==> word is not a pad word, i.e., a valid word
        mask_pre = tf.ones(shape=(tf.shape(mask)[0], 1))
        mask = tf.concat([mask_pre, mask[:, :-1]], axis=1)  # append a 1 to the mask for every batch, i.e., we cut off the start token before
        mask = tf.reshape(mask, [-1])

        output = - input * reward * mask  # per word: - (logprob * reward)
        output = tf.reduce_sum(output) / tf.reduce_sum(mask)
        avg_reward = tf.reduce_sum(reward) / tf.reduce_sum(mask)

        return output, avg_reward

    def get_reward(self, greedy_res, greedy_sample_res, video_ids, it=None):
        video_ids = [a.decode('utf-8') for a in video_ids.numpy()]

        gts = [self.imgs_gt[a] for a in video_ids]
        greedy_res_np = greedy_res.numpy()
        greedy_sample_res_np = greedy_sample_res.numpy()

        reward, cider_score = get_self_critical_reward(greedy_res_np, gts, greedy_sample_res_np, it=it)

        return reward, cider_score

    def train_step_rl(self, inputs):

        max_seq_len = tf.reduce_max(inputs['caption_len'] - 1, axis=0)
        target_ids = inputs[self.params['caption_id_feature_name']][:, 1:max_seq_len + 1]
        sample_n = 5

        if not self.params['use_i3d_instead_of_imgs']:
            image_features = inputs[self.params['image_feature_name']]
            frames_mask = inputs[self.params['frame_mask_feature_name']]
        else:
            image_features = inputs[self.params['i3d_feature_name']]
            frames_mask = inputs[self.params['i3d_frame_mask_feature_name']]

            if self.params['i3d_feats'] == 'features_i3d_spatial_rgb_imagenet':
                # shape [BS, -1, 7, 7, 1024] ==> [BS, -1, 1024]
                image_features = tf.reduce_mean(image_features, axis=[2,3])

        batch_size = tf.shape(image_features)[0]
        input_ids = tf.tile([self.params['start_id']], [batch_size])

        # Use greedy sampling with inference mode first (train=False)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(frames_mask, tf.expand_dims(input_ids, axis=1))

        if self.params['use_aud_feats']:
            audio_features = inputs['aud_features']
            aud_frames_mask = inputs['num_aud_frames_mask']
            aud_enc_padding_mask = create_padding_mask_frames(aud_frames_mask)

        if self.params['use_aud_feats']:
            gx, x, mask = self.model.encode_step(inp=image_features,
                                                 training=False,
                                                 enc_padding_mask=enc_padding_mask,
                                                 audio_inp=audio_features,
                                                 audio_mask=aud_enc_padding_mask,
                                                 i3d_timestamp_factor=inputs['i3d_timestamp_factor'],
                                                 aud_timestamp_factor=inputs['aud_timestamp_factor'])

            enc_output = (gx, x)
            audio_video_dec_padding_mask = mask
        else:
            gx, x, mask = self.model.encode_step(inp=image_features,
                                                 training=False,
                                                 enc_padding_mask=enc_padding_mask,
                                                 i3d_timestamp_factor=inputs['i3d_timestamp_factor'],
                                                 aud_timestamp_factor=inputs['aud_timestamp_factor'])

            enc_output = (gx, x)
            audio_video_dec_padding_mask = mask

        greedy_res, _ = self.greedy_sample(input_ids=input_ids,
                                           enc_output=enc_output,
                                           frames_mask=frames_mask,
                                           # combined_mask=combined_mask,
                                           dec_padding_mask=audio_video_dec_padding_mask,
                                           train_mode=False,
                                           sample_method="greedy")

        image_features = tf.repeat(image_features, axis=0, repeats=sample_n)
        i3d_timestamp_factor = tf.repeat(inputs['i3d_timestamp_factor'], axis=0, repeats=sample_n)
        frames_mask = tf.repeat(frames_mask, axis=0, repeats=sample_n)
        if self.params['use_aud_feats']:
            audio_features = tf.repeat(audio_features, axis=0, repeats=sample_n)
            aud_frames_mask = tf.repeat(aud_frames_mask, axis=0, repeats=sample_n)
            aud_frames_mask = create_padding_mask_frames(aud_frames_mask)
            aud_timestamp_factor =tf.repeat(inputs['aud_timestamp_factor'], axis=0, repeats=sample_n)

        # Now sample again with "train sampling"
        batch_size = tf.shape(image_features)[0]
        input_ids = tf.tile([self.params['start_id']], [batch_size])


        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(frames_mask, tf.expand_dims(input_ids, axis=1))

        with tf.GradientTape() as tape:
            if self.params['use_aud_feats']:
                gx, x, mask = self.model.encode_step(inp=image_features,
                                                    training=False,   # TODO: Why false?
                                                    enc_padding_mask=enc_padding_mask,
                                                    audio_inp=audio_features,
                                                    audio_mask=aud_frames_mask,
                                                    i3d_timestamp_factor=i3d_timestamp_factor,
                                                    aud_timestamp_factor=aud_timestamp_factor)
                enc_output = (gx, x)
                audio_video_dec_padding_mask = mask

            else:
                gx, x, mask = self.model.encode_step(inp=image_features,
                                                    training=False,   # TODO: Why false?
                                                    enc_padding_mask=enc_padding_mask,
                                                    i3d_timestamp_factor=inputs['i3d_timestamp_factor'],
                                                    aud_timestamp_factor=inputs['aud_timestamp_factor'])
                enc_output = (gx, x)
                audio_video_dec_padding_mask = mask

            greedy_sample_res, seq_logprobs = self.greedy_sample(input_ids=input_ids,
                                                                 enc_output=enc_output,
                                                                 frames_mask=frames_mask,
                                                                 #combined_mask=combined_mask,
                                                                 dec_padding_mask=audio_video_dec_padding_mask,
                                                                 sample_method="train",
                                                                 train_mode=True)


            greedy_res = greedy_res[:, 1:]
            greedy_sample_res = greedy_sample_res[:, 1:]
            reward, cider_score = tf.py_function(func=self.get_reward,
                                    inp=[greedy_res, greedy_sample_res, inputs['video_id'], self.optimizer.iterations],
                                    Tout=[tf.float32, tf.float32])

            seq = greedy_sample_res
            loss, avg_reward = self.reward_criterion(seq_logprobs, seq, reward)

        avg_greedy_sen_len = tf.reduce_mean(tf.reduce_sum(tf.cast(greedy_res > 0, dtype=tf.float32), axis=-1))
        avg_train_sen_len = tf.reduce_mean(tf.reduce_sum(tf.cast(greedy_sample_res > 0, dtype=tf.float32), axis=-1))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=0.1)



        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.global_norm.update_state(global_norm)
        self.avg_reward.update_state(avg_reward)
        self.greedy_sample_sentence_length.update_state(avg_greedy_sen_len)
        self.avg_train_ciderd.update_state(cider_score)
        self.train_sample_sentence_length.update_state(avg_train_sen_len)

        return loss

    def train_step(self, input):
        if self.model_type == 'default' or self.model_type == 'xla':
            return self.train_step_default(input)
        elif self.model_type == 'videobert':
            return self.train_step_videobert(input)
        elif self.model_type == 'LSTM':
            return self.train_step_lstm(input)


    def train_step_lstm(self, input):
        max_seq_len = tf.reduce_max(input['caption_len'] - 1, axis=0)
        input_ids = input[self.params['caption_id_feature_name']][:, :max_seq_len]
        target_ids = input[self.params['caption_id_feature_name']][:, 1:max_seq_len + 1]
        weights = input[self.params['caption_mask_feature_name']][:, :max_seq_len]

        with tf.GradientTape() as tape:
            lstm_output = self.model(inputs=input,
                                              input_ids=input_ids,
                                              max_time=max_seq_len,
                                              training=tf.constant(True, dtype=tf.bool),
                                              mask=weights)
            loss = self.loss_function(real=target_ids, pred=lstm_output) #, mask=weights)


        gradients = tape.gradient(loss, self.model.trainable_variables)

        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(target_ids, lstm_output)

        return loss


    def train_step_videobert(self, inp):
        max_seq_len = tf.reduce_max(inp['caption_len'] - 1, axis=0)
        input_ids = inp[self.params['caption_id_feature_name']][:, :max_seq_len]
        target_ids = inp[self.params['caption_id_feature_name']][:, 1:max_seq_len + 1]

        videobert_tokens = inp['videobert_tokens']
        videobert_tokens_mask = inp['videobert_tokens_mask']

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(videobert_tokens_mask, input_ids)



        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp=videobert_tokens,
                                            tar=input_ids,
                                            training=True,
                                            enc_padding_mask=enc_padding_mask,
                                            look_ahead_mask=combined_mask,
                                            dec_padding_mask=dec_padding_mask,)

            # TODO: not working? transformer loss from official package!
            loss = metrics.transformer_loss(predictions, target_ids,
                                           self.params["label_smoothing"],
                                           self.params["vocab_size"])

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(target_ids, predictions)

        return loss

    def train_step_default(self, inp):
        max_seq_len = tf.reduce_max(inp['caption_len'] - 1, axis=0)
        input_ids = inp[self.params['caption_id_feature_name']][:, :max_seq_len]
        target_ids = inp[self.params['caption_id_feature_name']][:, 1:max_seq_len + 1]

        if not self.params['use_i3d_instead_of_imgs']:
            image_features = inp[self.params['image_feature_name']]
            frames_mask = inp[self.params['frame_mask_feature_name']]
        else:
            image_features = inp[self.params['i3d_feature_name']]
            frames_mask = inp[self.params['i3d_frame_mask_feature_name']]

            if self.params['i3d_feats'] == 'features_i3d_spatial_rgb_imagenet':
                # shape [BS, -1, 7, 7, 1024] ==> [BS, -1, 1024]
                image_features = tf.reduce_mean(image_features, axis=[2,3])


        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(frames_mask, input_ids)

        if self.params['use_aud_feats']:
            audio_features = inp['aud_features']
            aud_frames_mask = inp['num_aud_frames_mask']
            aud_enc_padding_mask = create_padding_mask_frames(aud_frames_mask)


        with tf.GradientTape() as tape:
            if self.params['use_aud_feats']:
                predictions, _ = self.model.call_video_audio(
                    video_inp=image_features,
                    audio_inp=audio_features,
                    tar=input_ids,
                    training=True,
                    enc_padding_mask=enc_padding_mask,
                    enc_audio_padding_mask=aud_enc_padding_mask,
                    look_ahead_mask=combined_mask,
                    dec_padding_mask=dec_padding_mask,
                    i3d_timestamp_factor=inp['i3d_timestamp_factor'],
                    aud_timestamp_factor=inp['aud_timestamp_factor']
                )
            else:
                predictions, _ = self.model(inp=image_features,
                                            tar=input_ids,
                                            training=True,
                                            enc_padding_mask=enc_padding_mask,
                                            look_ahead_mask=combined_mask,
                                            dec_padding_mask=dec_padding_mask,
                                            i3d_timestamp_factor=inp['i3d_timestamp_factor'],
                                            aud_timestamp_factor=inp['aud_timestamp_factor']
                                            )

            loss = self.loss_function(target_ids, predictions)


        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(target_ids, predictions)

        return loss

    def calc_sent_stats(self, json_pred, max_num_sents):
        sent_distribution = [0] * max_num_sents
        unique_sents = []
        for i in range(max_num_sents):
            unique_sents.append(dict())
        for pred in json_pred:
            # sent_distribution[pred['num_sents'] - 1] += 1
            sents = nltk.sent_tokenize(pred['caption'])
            # assert (len(sents) == pred['num_sents'])
            for idx, sent in enumerate(sents):
                # unique_sent_for_sent_idx = unique_sents[idx]
                if idx in unique_sents:
                    if sent in unique_sents[idx]:
                        unique_sents[idx][sent] += 1
                    else:
                        unique_sents[idx][sent] = 1
        return sent_distribution, unique_sents

    def eval_performance(self, e, eval_ds_size, evalstep, test=False, custom_name=""):
        print(f"\nEvaluating performance at epoch {e + 1}")
        progbar = tf.keras.utils.Progbar(target=eval_ds_size)
        progbar.update(0)
        preds = []

        seen_image_ids = []

        for batch, sample in enumerate(self.dist_eval_dataset):
            res, video_ids = evalstep(sample)
            if type(self.strategy) is tf.distribute.MirroredStrategy:
                res = tf.concat(res.values, axis=0).numpy()
                video_ids = tf.concat(video_ids.values, axis=0).numpy()
            else:
                res = res.numpy()
                video_ids = video_ids.numpy()

            res = res[:, 1:]

            # Transform result to captions...
            for i in range(res.shape[0]):
                chars = []
                for char in res[i, :]:
                    if char == self.params['end_id']:
                        break
                    chars.append(char)
                image_id = video_ids[i]
                sentence = self.get_sentence(chars)

                if not image_id in seen_image_ids:
                    data = {
                        'image_id': image_id.decode('utf-8'),
                        'caption': sentence,
                    }
                    preds.append(data)
                    seen_image_ids.append(image_id)

            progbar.update(batch + 1)
        dir = './results/{}/'.format(os.path.basename(os.path.dirname(self.params['train_dir'])))

        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        prefix = ''
        if test:
            if custom_name != '':
                prefix = f"_TEST_{custom_name}"
            else:
                prefix = "_TEST_"

        fname = '{}captions_step_{}{}.json'.format(prefix, self.optimizer.iterations.numpy(), custom_name)
        full_name = os.path.join(dir, fname)
        with open(full_name, 'w') as f:
            json.dump(preds, f, indent=4)

        if custom_name != '':
            fname = 'vatex_submission_{}captions_step_{}{}.json'.format(prefix, self.optimizer.iterations.numpy(), custom_name)
            vatex_video_id_rgx = re.compile('video_VATEX_(?:private_test|test)_(.*_\d*_\d*)')
            vatex_submission = {}
            for single_pred in preds:
                m = vatex_video_id_rgx.match(single_pred['image_id'])
                if m:
                    vatex_submission[m.groups()[0]] = single_pred['caption']

            vatex_full_name = os.path.join(dir, fname)
            with open(vatex_full_name, 'w') as f:
                json.dump(vatex_submission, f, indent=4)

        if not self.coco is None:
            cocoEval = calculate_scores(fname=full_name,
                             coco=self.coco,
                             global_step=self.optimizer.iterations.numpy())

            self.save_best_checkpoint(cocoEval)

        sent_distribution, unique_sents = self.calc_sent_stats(preds, max_num_sents=1)
        for sentence_idx, distribution in enumerate(unique_sents):
            tf.summary.scalar(name='sent_stats/num_unique_sents_per_sent_idx_{}'.format(sentence_idx),
                              data=len(distribution),
                              step=self.optimizer.iterations.numpy())

    def create_tensorboard_vocab_info(self):
        offset = 1
        if self.params['use_subword_vocab'] != 'BERT':
            vocab_size = self.params['vocab_size'] - 2  # we dont need start and end token here
            if self.params['use_subword_vocab'] == 'True':
                offset = 0
        else:
            offset = 0
            vocab_size = self.params['vocab_size']

        with open(os.path.join(self.params['train_dir'], 'metadata.tsv'), "w") as f:
            for i in range(vocab_size):
                f.write("{}\n".format(self.dictionary[i + offset]))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "model/embedding_softmax_layer/weights/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.params['train_dir'], config)

    def try_init_from_checkpoint(self, init_from=None):
        with self.strategy.scope():
            # Restore augmenntor network
            if self.params['use_augmentor_network'] is not None:
                ckpt = tf.train.Checkpoint(mapping_network=self.model.mapping_network)
                path_augmn = self.params['use_augmentor_network']
                self.status_augmentor = ckpt.restore(path_augmn)
                # status.assert_existing_objects_matched()
                self.status_augmentor.expect_partial()
                print("Restored augmentor network from {}".format(path_augmn))

            # Restore embedding
            if self.params['init_word_embedding'] is not None:
                ckpt = tf.train.Checkpoint(embedding=self.model.embedding_softmax_layer)
                path_emb = self.params['init_word_embedding']
                self.status_we = ckpt.restore(path_emb)
                # status.assert_existing_objects_matched()
                self.status_we.expect_partial()
                print("Restored embeddings from {}".format(path_emb))

            # Restore encoder
            if self.params['init_video_encoder'] is not None:
                ckpt = tf.train.Checkpoint(embedding=self.model.encoder)
                path_emb = self.params['init_video_encoder']
                self.status_encoder = ckpt.restore(path_emb)
                #self.status_encoder.assert_existing_objects_matched()
                #self.status_encoder.assert_consumed()
                print("Restored video/audio encoder from {}".format(path_emb))

            # Restore videobert tokens
            if self.params['init_videobert'] is not None:
                ckpt = tf.train.Checkpoint(video_token_embedding=self.model.encoder.video_token_embedding)
                path_emb = self.params['init_videobert']
                self.status_encoder = ckpt.restore(path_emb)
                # self.status_encoder.assert_existing_objects_matched()
                # self.status_encoder.assert_consumed()
                print("Restored video token embeddings from {}".format(path_emb))
                pass

            self.ckpt = tf.train.Checkpoint(model=self.model,
                                       optimizer=self.optimizer)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.params['train_dir'], max_to_keep=100)

            # Restore existing checkpoint
            if not init_from is None:
                status = self.ckpt.restore(init_from)
                # status.assert_existing_objects_matched()
                status.expect_partial()
                print("Restored from {}".format(init_from))
                return True
            else:
                if self.ckpt_manager.latest_checkpoint:
                    status = self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                    # status.assert_existing_objects_matched()
                    # status.expect_partial()
                    print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
                    return True
                else:
                    print("Checkpoint not found!")
                    return False

    def save_checkpoint(self, report_iter):
        self.ckpt_manager.save(checkpoint_number=report_iter)

    def get_result_str(self, cocoEval, global_step):
        result_str = "\nCoco scores (Transformer Model; Best CIDEr @ iter {}): \n" \
                     "----------------------------------- \n\n".format(global_step)
        # print output evaluation scores
        for metric, score in cocoEval.eval.items():
            result_str += "{}:\t\t{:.4f}\n".format(metric, score)

        return result_str

    def save_best_pretrain_checkpoint(self, score):

        if score > self.max_pretrain_score:
            self.max_pretrain_score = score

            latest_checkpoint = self.ckpt_manager.latest_checkpoint
            ckpt_files = glob.glob(latest_checkpoint + '*')
            best_model_dir = os.path.join(self.params['train_dir'], 'best_model')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)

            global_step = self.optimizer.iterations.numpy()
            print("new best validation accuracy: {} @ iteration {}".format(self.max_pretrain_score, global_step))

            for file in ckpt_files:
                shutil.copy(file, best_model_dir + '/')

    def save_best_checkpoint(self, cocoEval):
        cider_score = cocoEval.eval['CIDEr']
        if cider_score > self.max_cider_score:
            self.max_cider_score = cider_score

            latest_checkpoint = self.ckpt_manager.latest_checkpoint
            ckpt_files = glob.glob(latest_checkpoint + '*')
            best_model_dir = os.path.join(self.params['train_dir'], 'best_model')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)

            global_step = self.optimizer.iterations.numpy()
            print("new best CIDEr score: {} @ iteration {}".format(self.max_cider_score, global_step))

            result_str = self.get_result_str(cocoEval, global_step)
            with open(os.path.join(best_model_dir, 'best_results_{}.txt'.format(global_step)), 'w') as f:
                f.write(result_str)

            for file in ckpt_files:
                shutil.copy(file, best_model_dir + '/')

    def get_sentence(self, chars):
        if self.params['use_subword_vocab'] == 'True':
            return self.tokenizer.decode(chars)
        elif self.params['use_subword_vocab'] == 'False':
            return " ".join([get_word(c, self.dictionary) for c in chars])
        elif self.params['use_subword_vocab'] == 'BERT':
            text = " ".join([get_word(c, self.dictionary) for c in chars])
            fine_text = text.replace(' ##', '')
            return fine_text