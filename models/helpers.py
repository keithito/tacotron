import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
  def __init__(self, batch_size, output_dim, r):
    with tf.name_scope('TacoTestHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim
      self._reduction_factor = r

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
    '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
    with tf.name_scope('TacoTestHelper'):
      # A sequence is finished when the stop token probability is > 0.5
      # With enough training steps, the model should be able to predict when to stop correctly
      # and the use of stop_at_any = True would be recommended. If however the model didn't
      # learn to stop correctly yet, (stops too soon) one could choose to use the safer option
      # to get a correct synthesis
      finished = tf.reduce_any(tf.cast(tf.round(stop_token_preds), tf.bool))

      # Feed last output frame as next input. outputs is [N, output_dim * r]
      next_inputs = outputs[:, -self._output_dim:]
      return (finished, next_inputs, state)


class TacoTrainingHelper(Helper):
  def __init__(self, inputs, targets, output_dim, r):
    # inputs is [N, T_in], targets is [N, T_out, D]
    with tf.name_scope('TacoTrainingHelper'):
      self._batch_size = tf.shape(inputs)[0]
      self._output_dim = output_dim
      self._reduction_factor = r

      # Feed every r-th target frame as input
      self._targets = targets[:, r-1::r, :]

      # Use full length for every target because we don't want to mask the padding frames
      num_steps = tf.shape(self._targets)[1]
      self._lengths = tf.tile([num_steps], [self._batch_size])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def token_output_size(self):
    return self._reduction_factor

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
    with tf.name_scope(name or 'TacoTrainingHelper'):
      finished = (time + 1 >= self._lengths)
      next_inputs = self._targets[:, time, :] # Teacher forcing: feed the true frame
      return (finished, next_inputs, state)


def _go_frames(batch_size, output_dim):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_size, output_dim])

