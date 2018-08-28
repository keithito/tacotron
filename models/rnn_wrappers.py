import collections
import numpy as np
import tensorflow as tf
from .modules import prenet
from .attention import _compute_attention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
from hparams import hparams as hp


class FrameProjection:
  """Projection layer to r * num_mels dimensions or num_mels dimensions
  """
  def __init__(self, shape=hp.num_mels, activation=None, scope=None):
    """
    Args:
      shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
      activation: callable, activation function
      scope: FrameProjection scope.
    """
    super(FrameProjection, self).__init__()

    self.shape = shape
    self.activation = activation
    self.scope = 'linear_projection' if scope is None else scope
    self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      # If activation==None, this returns a simple Linear projection
      # else the projection will be passed through an activation function
      # output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
      # name='projection_{}'.format(self.scope))
      return self.dense(inputs)


class StopProjection:
  """Projection to a scalar and through a sigmoid activation
  """
  def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
    """
    Args:
      is_training: Boolean, to control the use of sigmoid function as it is useless to use it
        during training since it is integrate inside the sigmoid_crossentropy loss
      shape: integer, dimensionality of output space. Defaults to 1 (scalar)
      activation: callable, activation function. only used during inference
      scope: StopProjection scope.
    """
    super(StopProjection, self).__init__()

    self.is_training = is_training
    self.shape = shape
    self.activation = activation
    self.scope = 'stop_token_projection' if scope is None else scope

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      output = tf.layers.dense(inputs, units=self.shape, activation=None, name='projection_{}'.format(self.scope))
      #During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
      return output if self.is_training else self.activation(output)


class TacotronDecoderCellState(
  collections.namedtuple("TacotronDecoderCellState",
   ("cell_state", "attention", "time", "alignments",
    "alignment_history"))):
  """`namedtuple` storing the state of a `TacotronDecoderCell`.
  Contains:
    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
    step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
     emitted at the previous time step for each attention mechanism.
    - `alignment_history`: a single or tuple of `TensorArray`(s)
     containing alignment matrices from all time steps for each attention
     mechanism. Call `stack()` on each to convert to a `Tensor`.
  """
  def replace(self, **kwargs):
    """Clones the current state while overwriting components provided by kwargs.
    """
    return super(TacotronDecoderCellState, self)._replace(**kwargs)


class TacotronDecoderWrapper(RNNCell):
  """Tactron 2 Decoder Cell
  Decodes encoder output and previous mel frames into next r frames

  Decoder Step i:
    1) Prenet to compress last output information
    2) Concat compressed inputs with previous context vector (input feeding) *
    3) Decoder RNN (actual decoding) to predict current state s_{i} *
    4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
    5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
    6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)

  * : This is typically taking a vanilla LSTM, wrapping it using tensorflow's attention wrapper,
  and wrap that with the prenet before doing an input feeding, and with the prediction layer
  that uses RNN states to project on output space. Actions marked with (*) can be replaced with
  tensorflow's attention wrapper call if it was using cumulative alignments instead of previous alignments only.
  """

  def __init__(self, is_training, attention_mechanism, rnn_cell, frame_projection, stop_projection):
    """Initialize decoder parameters

    Args:
        prenet: A tensorflow fully connected layer acting as the decoder pre-net
        attention_mechanism: A _BaseAttentionMechanism instance, usefull to
          learn encoder-decoder alignments
        rnn_cell: Instance of RNNCell, main body of the decoder
        frame_projection: tensorflow fully connected layer with r * num_mels output units
        stop_projection: tensorflow fully connected layer, expected to project to a scalar
          and through a sigmoid activation
      mask_finished: Boolean, Whether to mask decoder frames after the <stop_token>
    """
    super(TacotronDecoderWrapper, self).__init__()
    #Initialize decoder layers
    self._training = is_training
    self._attention_mechanism = attention_mechanism
    self._cell = rnn_cell
    self._frame_projection = frame_projection
    self._stop_projection = stop_projection
    self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
      self._attention_mechanism.batch_size,
      message=error_message)]

  @property
  def output_size(self):
    return self._frame_projection.shape

  # @property
  def state_size(self):
    """The `state_size` property of `TacotronDecoderWrapper`.

    Returns:
      An `TacotronDecoderWrapper` tuple containing shapes used by this object.
    """
    return TacotronDecoderCellState(
      cell_state=self._cell._cell.state_size,
      time=tensor_shape.TensorShape([]),
      attention=self._attention_layer_size,
      alignments=self._attention_mechanism.alignments_size,
      alignment_history=())

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `AttentionWrapper`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.
    Returns:
      An `TacotronDecoderCellState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.
    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
      `batch_size` does not match the output size of the encoder passed
      to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
        "When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the requested batch size.")
      with ops.control_dependencies(
        self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
          lambda s: array_ops.identity(s, name="checked_cell_state"),
          cell_state)
      return TacotronDecoderCellState(
        cell_state=cell_state,
        time=array_ops.zeros([], dtype=tf.int32),
        attention=rnn_cell_impl._zero_state_tensors(self._attention_layer_size, batch_size, dtype),
        alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
        alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
        dynamic_size=True))


  def __call__(self, inputs, state):
    #Information bottleneck (essential for learning attention)
    prenet_output = prenet(inputs, self._training, hp.prenet_depths, scope='decoder_prenet')

    #Concat context vector and prenet output to form RNN cells input (input feeding)
    rnn_input = tf.concat([prenet_output, state.attention], axis=-1)

    #Unidirectional RNN layers
    rnn_output, next_cell_state = self._cell(tf.layers.dense(rnn_input, hp.decoder_depth), state.cell_state)

    #Compute the attention (context) vector and alignments using
    #the new decoder cell hidden state as query vector
    #and cumulative alignments to extract location features
    #The choice of the new cell hidden state (s_{i}) of the last
    #decoder RNN Cell is based on Luong et Al. (2015):
    #https://arxiv.org/pdf/1508.04025.pdf
    previous_alignments = state.alignments
    previous_alignment_history = state.alignment_history
    context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
      rnn_output,
      previous_alignments,
      attention_layer=None)

    #Concat RNN outputs and context vector to form projections inputs
    projections_input = tf.concat([rnn_output, context_vector], axis=-1)

    #Compute predicted frames and predicted <stop_token>
    cell_outputs = self._frame_projection(projections_input)
    stop_tokens = self._stop_projection(projections_input)

    #Save alignment history
    alignment_history = previous_alignment_history.write(state.time, alignments)

    #Prepare next decoder state
    next_state = TacotronDecoderCellState(
      time=state.time + 1,
      cell_state=next_cell_state,
      attention=context_vector,
      alignments=cumulated_alignments,
      alignment_history=alignment_history)

    return (cell_outputs, stop_tokens), next_state
