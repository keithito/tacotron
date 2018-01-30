import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


#sentences = [
#  # From July 8, 2017 New York Times:
#  'Scientists at the CERN laboratory say they have discovered a new particle.',
#  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#  'President Trump met with other leaders at the Group of 20 conference.',
#  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
#  # From Google's Tacotron example page:
#  'Generative adversarial network or variational auto-encoder.',
#  'The buses aren\'t the problem, they actually provide a solution.',
#  'Does the quick brown fox jump over the lazy dog?',
#  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
#]

sentences = [
  "yi1 xiang1 gang3 de5 kang4 ri4 jiu4 wang2 yun4 dong4 yi3 jiu4 wang2 wen2 hua4 yun4 dong4 wei2 zhu3 yao4 xing2 shi4",
  'jv4 xi1 mei3 guo2 can1 iii4 vvvan4 iii3 cao2 ni3 iii1 fen4 zh ix1 ch ix2 k e4 l in2 d un4 x iang4 b o1 h ei1 p ai4 b ing1 d e5 j ve2 ii i4 aa an4 zh un3 b ei4 z ai4 b en3 vv ve4 sh ang4 x vn2 j in4 x ing2 b iao3 j ve2',
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
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
