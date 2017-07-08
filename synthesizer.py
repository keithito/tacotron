import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from models import create_model
from util import audio, textinput


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    seq = textinput.to_sequence(text,
      force_lowercase=hparams.force_lowercase,
      expand_abbreviations=hparams.expand_abbreviations)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    spec = self.session.run(self.model.linear_outputs[0], feed_dict=feed_dict)
    out = io.BytesIO()
    audio.save_wav(audio.inv_spectrogram(spec.T), out)
    return out.getvalue()
