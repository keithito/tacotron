import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import amy, blizzard, ljspeech, kusal, mailabs
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(
      in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
  in_dir = os.path.join(args.base_dir, 'LJSpeech-1.0')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = ljspeech.build_from_path(
      in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_amy(args):
  in_dir = os.path.join(args.base_dir, 'amy')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = amy.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_kusal(args):
  in_dir = os.path.join(args.base_dir, 'kusal')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = kusal.build_from_path(
      in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def preprocess_mailabs(args):
  in_dir = os.path.join(args.mailabs_books_dir)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  books = args.books
  metadata = mailabs.build_from_path(
      in_dir, out_dir, books, args.hparams, args.num_workers, tqdm)
  write_metadata(metadata, out_dir, args.hparams)


def write_metadata(metadata, out_dir, hparams=hparams):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' %
        (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--output', default='training')
  parser.add_argument(
      '--dataset', required=True, choices=['amy', 'blizzard', 'ljspeech', 'kusal', 'mailabs']
  )
  parser.add_argument('--mailabs_books_dir',
                      help='absolute directory to the books for the mlailabs')
  parser.add_argument(
      '--books',
      help='comma-seperated and no space name of books i.e hunter_space,pink_fairy_book,etc.',
  )
  parser.add_argument(
      '--hparams', default='',
      help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()

  if args.dataset == 'mailabs' and args.books is None:
    parser.error("--books required if mailabs is chosen for dataset.")

  if args.dataset == 'mailabs' and args.mailabs_books_dir is None:
    parser.error(
        "--mailabs_books_dir required if mailabs is chosen for dataset.")

  args.hparams = hparams.parse(args.hparams)

  if args.dataset == 'amy':
    preprocess_amy(args)
  elif args.dataset == 'blizzard':
    preprocess_blizzard(args)
  elif args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'kusal':
    preprocess_kusal(args)
  elif args.dataset == 'mailabs':
    preprocess_mailabs(args)


if __name__ == "__main__":
  main()
