import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMBlockCell


def prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i+1))
  return x


def conv_and_lstm(inputs, input_lengths, conv_layers, conv_width, conv_channels, lstm_units,
                  is_training, scope):
  # Convolutional layers
  with tf.variable_scope(scope):
    x = inputs
    for i in range(conv_layers):
      activation = tf.nn.relu if i < conv_layers - 1 else None
      x = conv1d(x, conv_width, conv_channels, activation, is_training, 'conv_%d' % i)

    # 2-layer bidirectional LSTM:
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      LSTMBlockCell(lstm_units),
      LSTMBlockCell(lstm_units),
      x,
      sequence_length=input_lengths,
      dtype=tf.float32,
      scope='encoder_lstm')

    # Concatentate forward and backwards:
    return tf.concat(outputs, axis=2)


def postnet(inputs, layers, conv_width, channels, is_training):
  x = inputs
  with tf.variable_scope('decoder_postnet'):
    for i in range(layers):
      activation = tf.nn.tanh if i < layers - 1 else None
      x = conv1d(x, conv_width, channels, activation, is_training, 'postnet_conv_%d' % i)
  return tf.layers.dense(x, inputs.shape[2])   # Project to input shape


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)
