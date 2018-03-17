import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention


class LocationSensitiveAttention(BahdanauAttention):
  '''Implements Location Sensitive Attention from:
  Chorowski, Jan et al. 'Attention-Based Models for Speech Recognition'
  https://arxiv.org/abs/1506.07503
  '''
  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               filters=20,
               kernel_size=7,
               name='LocationSensitiveAttention'):
    '''Construct the Attention mechanism. See superclass for argument details.'''
    super(LocationSensitiveAttention, self).__init__(
      num_units,
      memory,
      memory_sequence_length=memory_sequence_length,
      name=name)
    self.location_conv = tf.layers.Conv1D(
      filters, kernel_size, padding='same', use_bias=False, name='location_conv')
    self.location_layer = tf.layers.Dense(
      num_units, use_bias=False, dtype=tf.float32, name='location_layer')


  def __call__(self, query, state):
    '''Score the query based on the keys and values.
    This replaces the superclass implementation in order to add in the location term.
    Args:
      query: Tensor of shape `[N, num_units]`.
      state: Tensor of shape `[N, T_in]`
    Returns:
      alignments: Tensor of shape `[N, T_in]`
      next_state: Tensor of shape `[N, T_in]`
    '''
    with tf.variable_scope(None, 'location_sensitive_attention', [query]):
      expanded_alignments = tf.expand_dims(state, axis=2)               # [N, T_in, 1]
      f = self.location_conv(expanded_alignments)                       # [N, T_in, 10]
      processed_location = self.location_layer(f)                       # [N, T_in, num_units]

      processed_query = self.query_layer(query) if self.query_layer else query  # [N, num_units]
      processed_query = tf.expand_dims(processed_query, axis=1)         # [N, 1, num_units]
      score = _location_sensitive_score(processed_query, processed_location, self.keys)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def _location_sensitive_score(processed_query, processed_location, keys):
  '''Location-sensitive attention score function.
  Based on _bahdanau_score from tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
  '''
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  v = tf.get_variable('attention_v', [num_units], dtype=processed_query.dtype)
  return tf.reduce_sum(v * tf.tanh(keys + processed_query + processed_location), [2])
