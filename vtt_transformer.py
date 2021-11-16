import tensorflow as tf
import numpy as np
from official.nlp.transformer.transformer import *

from gan_augmentor_model import MappingNetwork
from official_transformer_vtt import TransformerVTT
from official.nlp.transformer.transformer_main import *
from official.nlp.transformer import misc
import tensorflow_addons as tfa
import sys

my_dtype = tf.float32

bert_tokens = {x: i for i, x in enumerate(['[CLS]', '[SPLIT]'])}

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        total_seq_len = tf.shape(scaled_attention_logits)[3]
        mask_seq_len = tf.shape(mask)[3]
        pad_right = total_seq_len - mask_seq_len
        mask = tf.pad(mask, [[0, 0], [0, 0], [0, 0], [0, pad_right]])
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
    ])

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def get_angles_tf(pos, i, d_model):
    angle_rates = 1 / tf.math.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=my_dtype)

def get_fractional_position_encoding(positions, d_model):
    d_model_range = tf.expand_dims(tf.range(d_model, dtype=tf.float32), axis=0)
    # get angle base vals
    angle_rads_tf = get_angles_tf(positions, d_model_range, d_model)

    # create indcs where to use sin or cos
    sin_idcs = tf.expand_dims(tf.range(0, d_model, 2), axis=0)
    cos_idcs = tf.expand_dims(tf.range(1, d_model, 2), axis=0)

    # Choose every element with x % 2 == 0 and apply sin and choose every element with (x+1) % 2 == 0 and apply cos
    tmp = tf.reduce_sum(tf.one_hot(sin_idcs, d_model), axis=1) * tf.cast(tf.sin(angle_rads_tf),
                                                                         dtype=my_dtype) + tf.reduce_sum(
        tf.one_hot(cos_idcs, d_model), axis=1) * tf.cast(tf.cos(angle_rads_tf), dtype=my_dtype)

    pos_encoding = tf.expand_dims(tmp, axis=0)
    return tf.cast(pos_encoding, dtype=my_dtype)

class GLULayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GLULayer, self).__init__()

    def call(self, x):
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.sigmoid(b)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, m=40):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.m = m

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        if m > 0:
            # memory
            m_k_initializer = tf.initializers.RandomNormal(mean=0, stddev=1. / d_model)
            self.m_k = tf.Variable(initial_value=m_k_initializer(shape=(1, m, d_model), dtype=tf.float32),
                                   trainable=True)

            m_v_initializer = tf.initializers.RandomNormal(mean=0, stddev=1. / m)
            self.m_v = tf.Variable(initial_value=m_v_initializer(shape=(1, m, d_model), dtype=tf.float32),
                                   trainable=True)
        #
        # def _glorot_initializer(fan_in, fan_out):
        #     limit = tf.math.sqrt(6.0 / (fan_in + fan_out))
        #     return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        # self.dense = tf.keras.layers.Dense(d_model,
        #                                    kernel_initializer=_glorot_initializer(512, 512),
        #                                    dtype=tf.float32)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wq(k)  # (batch_size, seq_len, d_model)
        v = self.wq(v)  # (batch_size, seq_len, d_model)


        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        #
        # k = tf.cast(k, dtype=tf.float32)
        # v = tf.cast(v, dtype=tf.float32)
        # q = tf.cast(q, dtype=tf.float32)

        if self.m > 0:
            # memory
            m_k = tf.reshape(self.m_k, (1, self.num_heads, self.m, self.depth)) # (1, 8, 40, 64)
            m_k = np.sqrt(self.d_model) * tf.tile(m_k, [batch_size, 1, 1, 1])
            m_v = tf.reshape(self.m_v, (1, self.num_heads, self.m, self.depth)) # (1, 8, 40, 64)
            m_v = np.sqrt(self.m) * tf.tile(m_v, [batch_size, 1, 1, 1])


            k = tf.concat([k, m_k], axis=2) # (batch_size, num_heads, seq_len_k + m, depth)
            v = tf.concat([v, m_v], axis=2) # (batch_size, num_heads, seq_len_v + m, depth)


        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention = tf.cast(scaled_attention, dtype=my_dtype)
        # attention_weights = tf.cast(attention_weights, dtype=my_dtype)


        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)
        #output = tf.cast(output, dtype=my_dtype)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, memory_vector_size, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, m=memory_vector_size)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

def celu(x, alpha=1.):
    return tf.nn.elu(x)
    #mask_greater = tf.cast()
    #return tf.maximum(0, x) + tf.minimum(0, alpha * (tf.exp(x/alpha)-1))

class XLABasicAtt(tf.keras.layers.Layer):
    def __init__(self, mid_dims, mid_dropout):
        super(XLABasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(tf.keras.layers.Dense(mid_dims[i], name=f"att_basic_{i}"))
            sequential.append(tf.keras.layers.ReLU())
            if mid_dropout > 0:
                sequential.append(tf.keras.layers.Dropout(mid_dropout))
        self.attention_basic = tf.keras.Sequential(layers=sequential) if len(sequential) > 0 else None
        self.attention_last = tf.keras.layers.Dense(mid_dims[-1], name=f"att_last")

    def call(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = tf.squeeze(attn_weights, axis=-1)
        if att_mask is not None:
            # att_mask should be [200, 50] --> expand to shape: [200, 1, 50] ???
            attn_weights = tf.where(tf.expand_dims(att_mask, axis=1) == 0, -1e9, attn_weights)
            #attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = tf.nn.softmax.softmax(attn_weights, axis=-1)

        attn = tf.matmul(tf.expand_dims(attn_weights, axis=-2), value2) #(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        attn = tf.squeeze(attn, axis=-2)
        return attn

class XLASCAtt(XLABasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(XLASCAtt, self).__init__(mid_dims, mid_dropout)


        self.attention_last = tf.keras.layers.Dense(1, name=f"scatt_att_last_1") #nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = tf.keras.layers.Dense(mid_dims[-1], name=f"scatt_att_last_2") #nn.Linear(mid_dims[-2], mid_dims[-1])

    def call(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        #attn_map = tf.reshape(attn_map, [batch_size, self.num_heads, -1, self.head_dim])
        if att_mask is not None:
            #att_mask = tf.squeeze(att_mask)
            att_mask = tf.expand_dims(att_mask, axis=1) #att_mask.unsqueeze(1)
            att_mask_ext = tf.expand_dims(att_mask, axis=-1) #att_mask.unsqueeze(-1)
            att_map_pool = tf.reduce_sum(att_map * att_mask_ext, axis=-2) / tf.reduce_sum(att_mask_ext, axis=-2)
        else:
            att_map_pool = tf.reduce_mean(att_map, axis=-2) #att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = tf.sigmoid(alpha_channel)

        alpha_spatial = tf.squeeze(alpha_spatial, axis=-1) #alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = tf.where(att_mask == 0, -1e9, alpha_spatial)
            #alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = tf.nn.softmax(alpha_spatial, axis=-1)

        if len(tf.shape(alpha_spatial)) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = tf.matmul(alpha_spatial, value2)  #torch.matmul(alpha_spatial, value2)
        else:
            value2 = tf.matmul(tf.expand_dims(alpha_spatial, axis=-2), value2) #torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)
            value2 = tf.squeeze(value2, axis=-2)

        attn = value1 * value2 * alpha_channel
        return attn

class XLAInProjLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads, **kwargs):
        super(XLAInProjLayer, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(output_dim, name=f"in_proj_dense")
        self.act = tf.keras.layers.Lambda(lambda x: celu(x)) #utils.activation(cfg.MODEL.BILINEAR.ACT)
        self.norm = tfa.layers.GroupNormalization(groups=num_heads)

    def call(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)

        return x

class XLALowRank(tf.keras.layers.Layer):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(XLALowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        output_dim = embed_dim #2 * embed_dim if cfg.MODEL.BILINEAR.ACT == 'GLU' else embed_dim

        # sequential = []
        # sequential.append(tf.keras.layers.Dense(output_dim, name=f"in_proj_q_dense")) # nn.Linear(embed_dim, output_dim))
        # act = tf.keras.layers.Lambda(lambda x: celu(x)) #utils.activation(cfg.MODEL.BILINEAR.ACT)
        # if act is not None:
        #     sequential.append(act)
        # sequential.append(tfa.layers.GroupNormalization(groups=self.num_heads))
        # self.in_proj_q = tf.keras.Sequential(layers=sequential)

        self.in_proj_q = XLAInProjLayer(output_dim=output_dim, num_heads=self.num_heads, name=f"in_proj_q_dense")
        self.in_proj_k = XLAInProjLayer(output_dim=output_dim, num_heads=self.num_heads, name=f"in_proj_k_dense")
        self.in_proj_v1 = XLAInProjLayer(output_dim=output_dim, num_heads=self.num_heads, name=f"in_proj_v1_dense")
        self.in_proj_v2 = XLAInProjLayer(output_dim=output_dim, num_heads=self.num_heads, name=f"in_proj_v2_dense")


        self.attn_net = XLASCAtt(att_mid_dim, att_mid_drop) #layers.create(att_type, att_mid_dim, att_mid_drop)
        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.buffer_keys = tf.zeros((batch_size, self.num_heads, 0, self.head_dim))
        self.buffer_value2 = tf.zeros((batch_size, self.num_heads, 0, self.head_dim))

    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    def call(self, query, key, mask, value1, value2):
        #tf.print("BLUBB", output_stream=sys.stdout)
        batch_size = tf.shape(query)[0]

        # = self.encoder_attn(query=gx,
        #                     key=x,
        #                     mask=mask,
        #                     value1=gx,
        #                     value2=x)

        #query = tf.reshape(query, [batch_size, -1, self.d_model])

        # x = tf.reshape(x, [batch_size, self.d_model])
        # gx = tf.reshape(gx, [batch_size, -1, self.d_model])
        # gx = tf.reshape(gx, [batch_size, -1, self.d_model])
        # gx = tf.reshape(gx, [batch_size, -1, self.d_model])
        #query = tf.reshape(query, [4, 512])
        #value1 = tf.reshape(value1, [4, 512])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = tf.reshape(q, shape=[batch_size, self.num_heads, self.head_dim])
        v1 = tf.reshape(v1, shape=[batch_size, self.num_heads, self.head_dim])

        # precompute = False
        #x = tf.reshape(x, [batch_size, self.d_model])
        #key = tf.reshape(key, [-1, tf.shape(key)[-1]])
        #value2 = tf.reshape(value2, [-1, tf.shape(value2)[-1]])
        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v2 = tf.reshape(v2, [batch_size, -1, self.num_heads, self.head_dim])
        v2 = tf.transpose(v2, [0, 2, 1, 3])


        attn_map = tf.expand_dims(q, axis=-2) * k
        attn_map = tf.reshape(attn_map, [batch_size, self.num_heads, -1, self.head_dim])
        mask = tf.reshape(mask, [batch_size, -1])
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = tf.reshape(attn, [batch_size, self.num_heads * self.head_dim])

        return attn
        # attn_map = q.unsqueeze(-2) * k
        # attn = self.attn_net(attn_map, mask, v1, v2)
        # attn = attn.view(batch_size, self.num_heads * self.head_dim)
        # return attn

    def call2(self, query, key, mask, value1, value2):
        batch_size = tf.shape(query)[0]
        #query = tf.reshape(query, [-1, tf.shape(query)[-1]]) #query.view(-1, query.size()[-1])
        #value1 = tf.reshape(value1, [-1, tf.shape(query)[-1]])

        query = tf.reshape(query, [-1, self.embed_dim])
        value1 = tf.reshape(value1, [-1, self.embed_dim])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = tf.reshape(q, shape=[batch_size, -1, self.num_heads, self.head_dim])
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        v1 = tf.reshape(v1, shape=[batch_size, -1, self.num_heads, self.head_dim])
        v1 = tf.transpose(v1, perm=[0, 2, 1, 3])

        # precompute = False
        # key = tf.reshape(key, [-1, tf.shape(key)[-1]])
        # value2 = tf.reshape(value2, [-1, tf.shape(value2)[-1]])
        key = tf.reshape(key, [-1, self.embed_dim])
        value2 = tf.reshape(value2, [-1, self.embed_dim])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v2 = tf.reshape(v2, [batch_size, -1, self.num_heads, self.head_dim])
        v2 = tf.transpose(v2, [0, 2, 1, 3])

        # if self.buffer_keys is not None and self.buffer_value2 is not None:
        #     self.buffer_keys = tf.concat([self.buffer_keys, k], axis=2) #torch.cat([self.buffer_keys, k], dim=2)
        #     self.buffer_value2 = tf.concat([self.buffer_value2, v2], axis=-2)
        #     k = self.buffer_keys
        #     v2 = self.buffer_value2

        attn_map = tf.expand_dims(q, axis=-2) * tf.expand_dims(k, axis=-3)

        num_q = tf.shape(q)[2]
        num_k = tf.shape(k)[2]
        attn_map = tf.reshape(attn_map, [batch_size, self.num_heads, num_q, num_k, self.head_dim])
        mask = tf.reshape(mask, [batch_size, -1, num_k])

        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])

        attn = tf.reshape(attn, [batch_size, -1, self.num_heads * self.head_dim])

        return attn
        # attn_map = q.unsqueeze(-2) * k
        # attn = self.attn_net(attn_map, mask, v1, v2)
        # attn = attn.view(batch_size, self.num_heads * self.head_dim)
        # return attn


class XLAEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, memory_vector_size, rate=0.1):
        super(XLAEncoderLayer, self).__init__()

        self.encoder_attn = XLALowRank(
            embed_dim=d_model,
            #att_type = att_type,
            att_heads=num_heads,
            att_mid_dim=[96, 48, 96], # ENCODE_ATT_MID_DIM: [96, 48, 96]
            att_mid_drop=rate) # ENCODE_ATT_MID_DROPOUT: 0.1

        self.dropout = tf.keras.layers.Dropout(rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = point_wise_feed_forward_network(d_model, dff) #ffn :      relu_dropout = ff_dropout, dropout = ff_dropout)

        self.bifeat_e = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, name=f"encoder_layer_bifeat_e_dense"), # 2 * d_model -> d_model
            tf.keras.layers.ReLU(), # BIFEAT_EMB_ACT: 'RELU'
            tf.keras.layers.Dropout(0.3) #ENCODE_BIFEAT_EMB_DROPOUT: 0.3
        ])


        # self.mha = MultiHeadAttention(d_model, num_heads, m=memory_vector_size)
        # self.ffn = point_wise_feed_forward_network(d_model, dff)
        #
        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, gx, x, mask, training):
        mask = tf.squeeze(mask)
        gx = self.encoder_attn(query=gx,
                               key=x,
                               mask=mask,
                               value1=gx,
                               value2=x)

        gx = self.dropout(gx, training=training)

        # gx should be [BS, dmodel], e.g., [200, 768]
        x_ = tf.broadcast_to(tf.expand_dims(gx, axis=1), tf.shape(x))
        x_ = tf.concat([x_, x], axis=-1)
        x = self.bifeat_e(x_) + x
        x = self.layer_norm(x)

        x = self.ffn(x)

        return gx, x
        # attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        #
        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        #
        # return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # v, k, q
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2

class XLADecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, last_layer=False):
        super(XLADecoderLayer, self).__init__()

        self.last_layer = last_layer
        self.word_attn = XLALowRank(
            embed_dim=d_model,
            att_heads=num_heads,
            att_mid_dim=[96, 48, 96], # DECODE_ATT_MID_DIM: [96, 48, 96],
            att_mid_drop=rate) # DECODE_ATT_MID_DROPOUT: 0.1
        self.word_dropout = tf.keras.layers.Dropout(rate)

        self.cross_att = XLALowRank(
            embed_dim=d_model,
            att_heads=num_heads,
            att_mid_dim=[96, 48, 96], # DECODE_ATT_MID_DIM: [96, 48, 96],
            att_mid_drop=rate) # DECODE_ATT_MID_DROPOUT: 0.1)

        self.cross_dropout = tf.keras.layers.Dropout(rate)
        self.layer_norm_cross = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if self.last_layer == False:
            self.bifeat_emb = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model, name=f"decoder_layer_bifeat_emb_dense"), # 2 * d_model -> d_model
                tf.keras.layers.ReLU(), # BIFEAT_EMB_ACT: 'RELU'
                tf.keras.layers.Dropout(0.3) #DECODE_BIFEAT_EMB_DROPOUT: 0.3
            ])
            self.layer_norm_x = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            self.ffn = point_wise_feed_forward_network(d_model,
                                                       dff)  # ffn :      relu_dropout = ff_dropout, dropout = ff_dropout)

        self.layer_norm_gx = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.mha1 = MultiHeadAttention(d_model, num_heads)
        # self.mha2 = MultiHeadAttention(d_model, num_heads)
        #
        # self.ffn = point_wise_feed_forward_network(d_model, dff)
        #
        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)
        # self.dropout3 = tf.keras.layers.Dropout(rate)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def call(self, gx, x, enc_output, training, look_ahead_mask, padding_mask):
        word_x = x
        residual = x
        x = self.word_attn.call2(
            query=gx,
            key=x,
            mask=look_ahead_mask,
            value1=gx,
            value2=x)
        x = self.word_dropout(x)
        x = residual + x

        residual = x
        x = self.layer_norm_cross(x)
        x = self.cross_att.call2(
            query=x,
            key=enc_output,#encoder_out if precompute == False else p_key,
            mask=padding_mask, #att_mask,
            value1=x,
            value2=enc_output, #encoder_out if precompute == False else p_value2,
            )
        x = self.cross_dropout(x)
        gx = residual + x
        gx = self.layer_norm_gx(gx)

        if self.last_layer == False:
            x_ = tf.concat([gx, word_x], axis=-1) # torch.cat([gx, word_x], dim = -1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)

            if self.ffn is not None:
                x = self.ffn(x)
        else:
            x = None

        return gx, x #attn_weights_block1, attn_weights_block2

        # # enc_output.shape == (batch_size, input_seq_len, d_model)
        #
        # attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        # attn1 = self.dropout1(attn1, training=training)
        # out1 = self.layernorm1(x + attn1)
        #
        # # v, k, q
        # attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
        #                                        padding_mask)  # (batch_size, target_seq_len, d_model)
        # attn2 = self.dropout2(attn2, training=training)
        # out2 = self.layernorm2(out1 + attn2)  # (batch_size, target_seq_len, d_model)
        #
        # ffn_output = self.ffn(out2)
        # ffn_output = self.dropout3(ffn_output, training=training)
        # out3 = self.layernorm3(out2 + ffn_output)
        #
        # return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        memory_vector_size=0,
                                        rate=rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class VideoTokenEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, memory_vector_size, params, rate=0.1):
        super(VideoTokenEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        #
        # self.image_embedding = tf.keras.layers.Dense(d_model)
        # self.audio_embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.video_token_embedding = embedding_layer.EmbeddingSharedWeights(
            params['videobert_size'], d_model) # 10 special chars for now...

        self.cls_layer = tf.keras.layers.Dense(units=2)

        self.enc_layers = [EncoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        memory_vector_size=memory_vector_size,
                                        rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, audio_x=None, audio_mask=None):
        # x.shape == (batch_size, num_frames, cnn_dim)
        # mask is a mask if at a certain index there is a frame
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding
        x = self.video_token_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, my_dtype))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # if not audio_x is None:
        #     audio_x = self.audio_embedding(audio_x)
        #     audio_x *= tf.math.sqrt(tf.cast(self.d_model, my_dtype))
        #     seq_len_audio = tf.shape(audio_x)[1]
        #     audio_pos_encoding = self.pos_encoding[:, 300:300+seq_len_audio, :]
        #
        #     audio_x += audio_pos_encoding
        #     audio_x = self.dropout(audio_x, training=training)
        #
        #     # concat audio and video!
        #
        #     cls_ids = tf.tile([bert_tokens['[CLS]']], [batch_size])
        #     sep_ids = tf.tile([bert_tokens['[SPLIT]']], [batch_size])
        #     cls = self.embedding_softmax_layer(cls_ids)
        #     cls = tf.expand_dims(cls, axis=1)
        #     sep = self.embedding_softmax_layer(sep_ids)
        #     sep = tf.expand_dims(sep, axis=1)
        #
        #     x = tf.concat([cls, x, sep, audio_x], axis=1)
        #
        #     # mask shape: TensorShape([32, 1, 1, 38])
        #     # audio_mask shape: TensorShape([32, 10])
        #     mask_for_seperators = tf.ones(shape=(batch_size, 1, 1, 1))
        #     mask = tf.concat([mask_for_seperators, mask, mask_for_seperators, audio_mask], axis=3)
        #

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding,
                 memory_vector_size, rate=0.1, xla=False, use_fractional_pe=False):
        super(ImageEncoder, self).__init__()

        self.xla = xla
        self.use_fractional_pe = use_fractional_pe
        self.d_model = d_model
        self.num_layers = num_layers

        self.image_embedding = tf.keras.layers.Dense(d_model)
        self.audio_embedding = tf.keras.layers.Dense(d_model)

        # Range for positional encoding if we use fractions for positional encodings
        self.d_model_range = tf.expand_dims(tf.range(d_model, dtype=tf.float32), axis=0)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            10, d_model) # 10 special chars for now...

        self.cls_layer = tf.keras.layers.Dense(units=2)

        if self.xla:
            self.enc_layers = [XLAEncoderLayer(d_model=d_model,
                                            num_heads=num_heads,
                                            dff=dff,
                                            memory_vector_size=memory_vector_size,
                                            rate=rate) for _ in range(num_layers)]
            self.proj_norm = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model, name=f"image_encoder_proj_norm_dense"),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ])
        else:
            self.enc_layers = [EncoderLayer(d_model=d_model,
                                            num_heads=num_heads,
                                            dff=dff,
                                            memory_vector_size=memory_vector_size,
                                            rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, audio_x=None, audio_mask=None, i3d_timestamp_factor=None, aud_timestamp_factor=None):
        # x.shape == (batch_size, num_frames, cnn_dim)
        # mask is a mask if at a certain index there is a frame
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding
        x = self.image_embedding(x)  # (batch_size, num_frames??, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, my_dtype))

        if self.use_fractional_pe:
            fractional_pe = self.compute_fractional_pe_for_seq(batch_size, i3d_timestamp_factor, seq_len)
            # add those fractional pe on top
            x += fractional_pe
        else:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        if not audio_x is None:
            audio_x = self.audio_embedding(audio_x)
            audio_x *= tf.math.sqrt(tf.cast(self.d_model, my_dtype))
            seq_len_audio = tf.shape(audio_x)[1]

            if self.use_fractional_pe:
                fractional_pe_aud = self.compute_fractional_pe_for_seq(batch_size, aud_timestamp_factor, seq_len_audio)
                # add those fractional pe on top
                audio_x += fractional_pe_aud
            else:
                audio_pos_encoding = self.pos_encoding[:, 300:300+seq_len_audio, :]
                audio_x += audio_pos_encoding

            audio_x = self.dropout(audio_x, training=training)

            # concat audio and video!

            cls_ids = tf.tile([bert_tokens['[CLS]']], [batch_size])
            sep_ids = tf.tile([bert_tokens['[SPLIT]']], [batch_size])
            cls = self.embedding_softmax_layer(cls_ids)
            cls = tf.expand_dims(cls, axis=1)
            sep = self.embedding_softmax_layer(sep_ids)
            sep = tf.expand_dims(sep, axis=1)

            x = tf.concat([cls, x, sep, audio_x], axis=1)

            # mask shape: TensorShape([32, 1, 1, 38])
            # audio_mask shape: TensorShape([32, 10])
            mask_for_seperators = tf.ones(shape=(batch_size, 1, 1, 1))
            mask = tf.concat([mask_for_seperators, mask, mask_for_seperators, audio_mask], axis=3)

        if self.xla:
            #mask expected shape: [BS, 50], x expected shape: [BS, 50, 512/768]
            # tf.expand_dims(mask, axis=-1) --> [BS, 50, 1]
            m = tf.squeeze(mask)
            m = tf.expand_dims(m, axis=-1)
            #tf.print(x, output_stream=sys.stdout)
            gx = tf.reduce_sum(x * m, axis=1) / tf.reduce_sum(m, axis=1)

            # x = tf.reshape(x, [batch_size, self.d_model])
            # gx = tf.reshape(gx, [batch_size, -1, self.d_model])
            x = tf.reshape(x, [batch_size, -1, self.d_model])
            gx = tf.reshape(gx, [batch_size, self.d_model])

            gx_arr = [gx]
            #tf.print(mask, output_stream=sys.stdout)

            #for loop_idx in tf.range(start=0, limit=self.num_layers, delta=1):
            for i in range(self.num_layers):
                # (self, gx, x, mask, training):
                gx, x = self.enc_layers[i](gx, x, mask, training)
                gx_arr.append(gx)

            gx = tf.concat(gx_arr, axis=-1)
            gx = self.proj_norm(gx)

            return gx, x, mask
        else:
            for i in range(self.num_layers):
                x = self.enc_layers[i](x, training, mask)

            return None, x, mask  # (batch_size, input_seq_len, d_model)

    def compute_fractional_pe_for_seq(self, batch_size, timestamp_factor, seq_len):
        # get frame idcs int-wise, i.e., [0,1,2,...x] batch_size times ==> [batchsize, seq_len]
        frame_int_idcs = tf.reshape(tf.tile(tf.range(seq_len, dtype=tf.float32), [batch_size]), (batch_size, -1))
        # get i3d timestamp factor to match int idcs
        timestamp_factor = tf.expand_dims(timestamp_factor, axis=1)
        # Get fractional idcs
        fractional_positions = tf.expand_dims(frame_int_idcs * timestamp_factor, axis=-1)
        # Now calculate the fractional pe
        fractional_pe = get_fractional_position_encoding(fractional_positions, self.d_model)
        fractional_pe = tf.squeeze(fractional_pe, axis=0)
        return fractional_pe


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1, xla=False):
        super(Decoder, self).__init__()
        self.xla = xla
        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        #self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        if self.xla:
            self.dec_layers = [XLADecoderLayer(d_model=d_model,
                                            num_heads=num_heads,
                                            dff=dff,
                                            rate=rate,
                                            last_layer = (i == num_layers -1)) for i in range(num_layers)]


            self.wbil1 = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model, name=f"decoder_wbil1_dense"),
                tf.keras.layers.Lambda(lambda x: celu(x)),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ])

            self.wbil2 = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model, name=f"decoder_wbil2_dense"),
                tf.keras.layers.Lambda(lambda x: celu(x)),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ])

            self.wbi_drop = tf.keras.layers.Dropout(0.5)  # DECODE_DROPOUT: 0.5
            self.dropout_lm = tf.keras.layers.Dropout(0.5)  # DROPOUT_LM: 0.5

            # TODO GLU
            self.proj_norm = tf.keras.Sequential([
                tf.keras.layers.Dense(2*d_model, name=f"decoder_proj_norm_dense"),
                GLULayer(),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ])
        else:
            self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def init_buffer(self, batch_size):
        if self.xla:
            self.seq_len = 0
            self.x = tf.zeros((batch_size, 1, self.d_model))
            for layer in self.dec_layers:
                layer.init_buffer(batch_size)

    def clear_buffer(self):
        if self.xla:
            self.seq_len = None
            self.x = None
            for layer in self.layers:
                layer.clear_buffer()

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, gx=None):
        batch_size = tf.shape(x)[0]
        if self.xla:
            # look_ahead_mask = seq_mask,
            # padding_mask = att_mask
            att_mask = tf.expand_dims(tf.squeeze(padding_mask), axis=1)

            # already embedded?
            seq_len = tf.shape(x)[1]

            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            # TODO: layernorm word in original code

            if self.dropout is not None:
                x = self.dropout(x, training=training)

            # decoder layers
            gx = self.wbil1(gx)
            seq_mask = tf.squeeze(1. - look_ahead_mask)

            x_gx = (tf.reduce_sum(tf.expand_dims(x, axis=1) * tf.expand_dims(seq_mask, axis=-1), -2)
                    / tf.expand_dims(tf.reduce_sum(seq_mask, axis=-1), axis=-1))

            x_gx = tf.reshape(x_gx, tf.shape(x))

            x_gx = self.wbil2(x_gx)
            gx = tf.expand_dims(gx, axis=1) #gx.unsqueeze(1)
            gx = gx * x_gx
            gx = self.wbi_drop(gx, training=training)

            x = tf.reshape(x, [batch_size, -1, self.d_model])
            gx = tf.reshape(gx, [batch_size, -1, self.d_model]) # TODO: Shape ok??? bei encoder: [4,  512])

            gx_arr = [gx]
            for i in range(self.num_layers):
                #if precompute == False:
                #p_key = None
                #p_value2 = None
                #else:
                #    p_key, p_value2 = p_att_feats[layerid]
                gx, x = self.dec_layers[i](
                    gx=gx,
                    x=x,
                    enc_output=enc_output,
                    padding_mask=att_mask,
                    look_ahead_mask=seq_mask)
                gx_arr.append(gx)

            gx = tf.concat(gx_arr, axis=-1)
            gx = self.proj_norm(gx)

            gx = self.dropout_lm(gx, training=training)
            #out = self.generator(gx)
            return gx, {}

        else:
            # att_mask:=

            seq_len = tf.shape(x)[1]
            attention_weights = {}

            #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
            #x = tf.cast(x, dtype=tf.float32)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            #x += self.pos_encoding[:, :seq_len, :]

            #x = self.dropout(x, training=training)

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

                attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
                attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

            # x.shape == (batch_size, target_seq_len, d_model)
            return x, attention_weights

class ImageCaptioningTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, pe_input, pe_target,
                 memory_vector_size, params, rate=0.1):
        super(ImageCaptioningTransformer, self).__init__()

        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            target_vocab_size, d_model)
        self.rate = rate
        self.params = params
        if not params['train_word_embedding']:
            self.embedding_softmax_layer.trainable = False

        self.use_augmentor_network = params['use_augmentor_network']
        if self.use_augmentor_network is not None:
            self.mapping_network = MappingNetwork()
            self.mapping_network.trainable = False

        self.position_embedding = position_embedding.RelativePositionEmbedding(
            hidden_size=d_model)


        if self.params['model_type'] == 'default':
            self.encoder = ImageEncoder(num_layers=num_layers,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        maximum_position_encoding=pe_input,
                                        memory_vector_size=memory_vector_size,
                                        rate=rate,
                                        use_fractional_pe=params['use_timestamp_factors'])

        elif self.params['model_type'] == 'xla':
                self.encoder = ImageEncoder(num_layers=num_layers,
                                            d_model=d_model,
                                            num_heads=num_heads,
                                            dff=dff,
                                            maximum_position_encoding=pe_input,
                                            memory_vector_size=memory_vector_size,
                                            rate=rate,
                                            xla=True,
                                            use_fractional_pe=params['use_timestamp_factors'])
        elif self.params['model_type'] == 'videobert':
            self.encoder = VideoTokenEncoder(num_layers=num_layers,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        maximum_position_encoding=pe_input,
                                        memory_vector_size=memory_vector_size,
                                        rate=rate,
                                             params=params)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate, xla=self.params['model_type'] == 'xla')

        #self.final_layer = tf.keras.layers.Dense(target_vocab_size) # float16: , dtype=tf.float32)

    def encode_step(self, inp, training, enc_padding_mask, audio_inp=None, audio_mask=None,
                    i3d_timestamp_factor=None, aud_timestamp_factor=None):

        # inp.shape == (batch_size, num_frames, cnn_dim)
        if self.use_augmentor_network is not None and training:
            rnd = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.dtypes.float32)
            if rnd > 0.:
                bs = tf.shape(inp)[0]
                num_frames = tf.shape(inp)[1]
                scale, shift = self.mapping_network(tf.random.truncated_normal(shape=(bs, 128)))

                # expand dims on the num_frames axis, so the same 'augmentation' is used for every frame wihtin a single video
                # expected shape (BS, 2048), but inp is (BS, num_frames, 2048)
                # therefore expand scale, shift to (BS, 1, 2048), then everything should be broadcastet correctly
                scale = tf.expand_dims(scale, axis=1)
                shift = tf.expand_dims(shift, axis=1)

                inp = inp * scale + shift

        #tf.print(enc_padding_mask, output_stream=sys.stdout)
        enc_output = self.encoder(x=inp,
                                  training=training,
                                  mask=enc_padding_mask,
                                  audio_x=audio_inp,
                                  audio_mask=audio_mask,
                                  i3d_timestamp_factor=i3d_timestamp_factor,
                                  aud_timestamp_factor=aud_timestamp_factor)  # (batch_size, inp_seq_len, d_model)

        return enc_output

    def decode_step(self, enc_output, tar, training, look_ahead_mask, dec_padding_mask):
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(tar)
            decoder_inputs = tf.cast(decoder_inputs, my_dtype)

            with tf.name_scope("add_pos_encoding"):
                pos_encoding = self.position_embedding(decoder_inputs)
                pos_encoding = tf.cast(pos_encoding, my_dtype)
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.rate)

        if self.params['model_type'] == 'xla':
            gx, enc_output = enc_output
            dec_output, attention_weights = self.decoder(decoder_inputs, enc_output, training, look_ahead_mask,
                                                         dec_padding_mask, gx=gx)
        else:
            _, enc_output = enc_output
            dec_output, attention_weights = self.decoder(decoder_inputs, enc_output, training, look_ahead_mask,
                                                         dec_padding_mask)
        logits = self.embedding_softmax_layer(dec_output, mode="linear")
        logits = tf.cast(logits, tf.float32)

        return logits, attention_weights


    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, i3d_timestamp_factor, aud_timestamp_factor):

        gx, x, mask = self.encode_step(inp=inp,
                                       training=training,
                                       enc_padding_mask=enc_padding_mask,
                                       i3d_timestamp_factor=i3d_timestamp_factor,
                                       aud_timestamp_factor=aud_timestamp_factor)
        enc_output = (gx, x)
        dec_padding_mask = mask

        logits, attention_weights = self.decode_step(enc_output, tar, training, look_ahead_mask, dec_padding_mask)

        return logits, attention_weights

    def call_video_audio(self,
                         video_inp,
                         audio_inp,
                         tar,
                         training,
                         enc_padding_mask,
                         enc_audio_padding_mask,
                         look_ahead_mask,
                         dec_padding_mask,
                         i3d_timestamp_factor,
                         aud_timestamp_factor):
        #tf.print(tf.constant([1,2]), output_stream=sys.stdout)

        #tf.print(enc_padding_mask, output_stream=sys.stdout)
        gx, x, mask = self.encode_step(inp=video_inp,
                                      training=training,
                                      enc_padding_mask=enc_padding_mask,
                                      audio_inp=audio_inp,
                                      audio_mask=enc_audio_padding_mask,
                                      i3d_timestamp_factor=i3d_timestamp_factor,
                                      aud_timestamp_factor=aud_timestamp_factor)
        enc_output = (gx, x)
        audio_video_dec_padding_mask = mask


        logits, attention_weights = self.decode_step(enc_output, tar, training, look_ahead_mask, audio_video_dec_padding_mask) #dec_padding_mask)




        return logits, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

