import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
import time


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron',
           inter_num_threads=None, intra_num_threads=None):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)

    print('Loading checkpoint: %s' % checkpoint_path)
    if inter_num_threads != None and intra_num_threads != None:
      session_config = tf.ConfigProto(
        inter_op_parallelism_threads=inter_num_threads,
        intra_op_parallelism_threads=intra_num_threads)
    else:
      session_config = tf.ConfigProto()
    self.session = tf.Session(config=session_config)
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, iter_curr, iter_total):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    session_start = time.time()
    linear_outputs = self.session.run(self.model.linear_outputs,
                                      feed_dict=feed_dict)
    session_end = time.time()
    elapsed_runtime = session_end - session_start

    return linear_outputs[0], elapsed_runtime
