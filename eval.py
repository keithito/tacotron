import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


# sentences = [
#   # From July 8, 2017 New York Times:
#   'Scientists at the CERN laboratory say they have discovered a new particle.',
#   'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#   'President Trump met with other leaders at the Group of 20 conference.',
#   'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
#   # From Google's Tacotron example page:
#   'Generative adversarial network or variational auto-encoder.',
#   'The buses aren\'t the problem, they actually provide a solution.',
#   'Does the quick brown fox jump over the lazy dog?',
#   'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
# ]

sentences = [
  'ta1 jing3 ti4 de5 xia4 le5 chuang2 gei3 liang3 ge5 sun1 zi5 ye4 hao3 bei4 zi5 you4 na2 guo4 yi1 ba3 da4 yi3 zi5 ba3 jie3 mei4 lia3 dang3 zhu4 gang1 zou3 dao4 ke4 ting1 jiu4 bei4 ren2 lan2 yao1 bao4 zhu4 le5',
  'wei1 xin4 zhi1 fu4 zhang1 xiao3 long2 han3 jian4 lou4 mian4 cheng1 wei1 xin4 bu4 hui4 cha2 kan4 yong4 hu4 liao2 tian1 ji4 lu4 yi4 si an4 feng4 zhi1 fu4 bao3 , ben3 wen2 lai2 zi4 teng2 xun4 ke1 ji4 .',
  'da4 hui4 zhi3 re4 nao5 tou2 liang3 tian1 yue4 hou4 yue4 song1 kua3 zui4 zhong1 chu1 ben3 lun4 wen2 ji2 jiu4 suan4 yuan2 man3 wan2 cheng2 ren4 wu5',
  'lian2 dui4 zhi3 liu2 xia4 yi4 ming2 zhi2 ban1 yuan2 chui1 shi4 yuan2 si4 yang3 yuan2 wei4 sheng1 yuan2 deng3 ye3 lie4 dui4 pao3 bu4 gan2 wang3 zai1 qu1',
  'yi1 jiu3 wu3 ling2 nian2 ba1 yue4 zhong1 yang1 ren2 min2 zheng4 fu3 zheng4 wu4 yuan4 ban1 bu4 le5 bao3 zhang4 fa1 ming2 quan2 yu3 zhuan1 li4 quan2 zan4 xing2 tiao2 li4',
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
    path = '%s-%03d.wav' % (base_path, i)
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
