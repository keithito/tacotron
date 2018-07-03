import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import plot


sentences = [
    # From July 8, 2017 New York Times:
    # 'Scientists at the CERN laboratory say they have discovered a new particle.',
    # 'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
    # 'President Trump met with other leaders at the Group of 20 conference.',
    # 'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    # # From Google's Tacotron example page:
    # 'Generative adversarial network or variational auto-encoder.',
    # 'The buses aren\'t the problem, they actually provide a solution.',
    # 'Does the quick brown fox jump over the lazy dog?',
    # 'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
    "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    "Be a voice, not an echo.",
    "The human voice is the most perfect instrument of all.",
    "I'm sorry Dave, I'm afraid I can't do that.",
    "This cake is great, It's so delicious and moist.",
    "hello my name is mycroft.",
    "hi.",
    "wow.",
    "cool.",
    "great.",
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    wav_path = '%s-%d.wav' % (base_path, i)
    align_path = '%s-%d.png' % (base_path, i)
    print('Synthesizing and plotting: %s' % wav_path)
    wav, alignment = synth.synthesize(text)
    with open(wav_path, 'wb') as f:
      f.write(wav)
    plot.plot_alignment(
        alignment, align_path,
        info='%s' % (text)
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True,
                      help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
                      help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--force_cpu', default=False,
                      help='Force synthesize with cpu')
  args = parser.parse_args()
  if args.force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
