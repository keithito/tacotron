import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper



class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
    '''Initializes the model for inference.

    Sets "mel_outputs", "linear_outputs", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
    '''
    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.get_variable(
        'embedding', [len(symbols), 256], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)           # [N, T_in, 256]

      # Encoder
      prenet_outputs = prenet(embedded_inputs, is_training)                       # [N, T_in, 128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training)  # [N, T_in, 256]

      # Attention
      attention_cell = AttentionWrapper(
        DecoderPrenetWrapper(GRUCell(256), is_training),
        BahdanauAttention(256, encoder_outputs),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, 256]

      # Concatenate attention context vector and RNN cell output into a 512D vector.
      concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              # [N, T_in, 512]

      # Decoder (layers specified bottom to top):
      decoder_cell = MultiRNNCell([
          OutputProjectionWrapper(concat_cell, 256),
          ResidualWrapper(GRUCell(256)),
          ResidualWrapper(GRUCell(256))
        ], state_is_tuple=True)                                                  # [N, T_in, 256]

      # Project onto r mel spectrograms (predict r outputs at each RNN step):
      output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      if is_training:
        helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
      else:
        helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

      (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell, helper, decoder_init_state),
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]

      # Reshape outputs to be one output per entry
      mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels]) # [N, T_out, M]

      # Add post-processing CBHG:
      post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training)           # [N, T_out, 256]
      linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)               # [N, T_out, F]

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.mel_outputs = mel_outputs
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      log('Initialized Tacotron model. Dimensions: ')
      log('  embedding:               %d' % embedded_inputs.shape[-1])
      log('  prenet out:              %d' % prenet_outputs.shape[-1])
      log('  encoder out:             %d' % encoder_outputs.shape[-1])
      log('  attention out:           %d' % attention_cell.output_size)
      log('  concat attn & out:       %d' % concat_cell.output_size)
      log('  decoder cell out:        %d' % decoder_cell.output_size)
      log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
      log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
      log('  postnet out:             %d' % post_outputs.shape[-1])
      log('  linear out:              %d' % linear_outputs.shape[-1])


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      l1 = tf.abs(self.linear_targets - self.linear_outputs)
      # Prioritize loss for frequencies under 3000 Hz.
      n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
      self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
      self.loss = self.mel_loss + self.linear_loss


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
