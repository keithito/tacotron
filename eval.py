import argparse
import os
import re
from hparams import hparams, hparams_debug_string


sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',  # 5s
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  print(args)
  base_path = get_output_base_path(args.checkpoint)

  if not args.benchmark_only:
    # default, run with Grifflim-Lim layers to generate wav
    from synthesizer import Synthesizer
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
      path = '%s-%d.wav' % (base_path, i)
      print('Synthesizing: %s' % path)
      with open(path, 'wb') as f:
        f.write(synth.synthesize(text))
  else:
    # for benchmark-only, run without Grifflim-Lim layers
    # to generate mel spectrum
    from synthesizer_mel_spectrum import Synthesizer
    steps = args.steps
    warmup_steps = args.warmup_steps
    total_time = 0
    mel_spectrum_len = 5

    synth = Synthesizer()
    synth.load(args.checkpoint,
               inter_num_threads=args.num_inter_threads,
               intra_num_threads=args.num_intra_threads)

    for i in range(steps):
      # select 1st str to generate 5s mel spectrum
      text = sentences[0]
      result, elapsed_time = synth.synthesize(text, i, steps)
      if (i + 1) % 10 == 0:
        print("steps = {0}, {1} ms, {2} s mel_spectrum/sec"
              .format(i + 1, str(elapsed_time * 1000), 
                      str(mel_spectrum_len / elapsed_time)))
      # collect session run time during middle iterations,
      # since performance may fluctuate at the beginning and ending.
      if warmup_steps < i + 1 <= steps * 0.9:
        total_time += elapsed_time

    if steps >= 10:
      eval_interations = int(steps * 0.9) - warmup_steps
      time_average = total_time / eval_interations
      # mel spectrum len: able to generate how much wave using the mel spectrum 
      print('Mel spectrum len: {0} s'.format(str(mel_spectrum_len)))
      # latency: cost how much session run time to generate 5s mel spectrum
      print('Latency: {0:.4f} ms'.format(time_average * 1000))
      # throughput: generate how much mel spectrum per second
      print('Throughput: {0:.4f} s mel_spectrum/sec'.format(mel_spectrum_len / time_average))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--benchmark-only',
                      help='Choose to only measure tacotron performance'
                           'without Grifflim-Lim layers',
                      action='store_true', dest='benchmark_only')
  parser.add_argument("--warmup-steps",
                      help="The number of warmup steps",
                      dest='warmup_steps', type=int, default=10)
  parser.add_argument("--steps", type=int, default=100,
                      help="The number of steps")
  parser.add_argument('-e', "--num-inter-threads",
                      help='The number of inter-thread.',
                      dest='num_inter_threads', type=int, default=0)
  parser.add_argument('-a', "--num-intra-threads",
                      help='The number of intra-thread.',
                      dest='num_intra_threads', type=int, default=0)
  args = parser.parse_args()
  # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
