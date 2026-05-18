import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import numpy as np

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = Dense(self.head_dim, use_bias=False)
        self.keys = Dense(self.head_dim, use_bias=False)
        self.queries = Dense(self.head_dim, use_bias=False)
        self.fc_out = Dense(embed_size)

    def call(self, values, keys, query, mask):
        N = tf.shape(query)[0]
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]

        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
        if mask is not None:
            energy = tf.where(mask == 0, float("-1e20"), energy)

        attention = tf.nn.softmax(energy / tf.math.sqrt(tf.cast(self.embed_size, tf.float32)), axis=3)

        out = tf.einsum("nhql,nlhd->nqhd", attention, values)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))

        out = self.fc_out(out)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, max_len, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

        # Define eigenvector matrix for positional encoding
        self.positional_encoding = tf.Variable(tf.random.normal((max_len, embed_size), stddev=0.02))

    def call(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        attention = self.dropout(attention)
        attention = self.norm1(attention + query)

        # Add eigenvector positional encoding
        query += tf.nn.embedding_lookup(self.positional_encoding, tf.range(tf.shape(query)[1]))

        return attention


def create_transformer_model(input_shape, num_classes, num_layers=6, heads=8, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    # Convolutional Layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)

    # Determine the embed_size based on the output shape of the convolutional layers
    embed_size = x.shape[-1]

    # Define maximum sequence length
    max_len = input_shape[0]

    # Transformer Layers
    for _ in range(num_layers):
        x = TransformerBlock(embed_size, heads, max_len, dropout)(x, x, x, None)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Output layer
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model