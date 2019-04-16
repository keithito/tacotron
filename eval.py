import argparse
import os
import re
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(ckpt_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(ckpt_dir):
  print(hparams_debug_string())
  checkpoint = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
  synth = Synthesizer()
  synth.load(checkpoint)
  base_path = get_output_base_path(checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%03d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='logs-tacotron', help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  hparams.parse(args.hparams)
  run_eval(args.checkpoint)


if __name__ == '__main__':
  main()
