from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import librosa
import numpy as np
import os

from hparams import hparams
from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the Amy dataset from a given input path into a given output directory.'''
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []

  # Read all of the .wav files:
  paths = {}
  for path in glob.glob(os.path.join(in_dir, 'audio', '*.wav')):
    prompt_id = os.path.basename(path).split('-')[-2]
    paths[prompt_id] = path

  # Read the prompts file:
  with open(os.path.join(in_dir, 'prompts.txt'), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('\t')
      if len(parts) == 3 and parts[0] in paths:
        path = paths[parts[0]]
        text = parts[2]
        futures.append(executor.submit(partial(_process_utterance, out_dir, parts[0], path, text)))
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, prompt_id, wav_path, text):
  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Trim leading and trailing silence:
  margin = int(hparams.sample_rate * 0.1)
  wav = wav[margin:-margin]
  wav, _ = librosa.effects.trim(wav, top_db=40, frame_length=1024, hop_length=256)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'amy-spec-%s.npy' % prompt_id
  mel_filename = 'amy-mel-%s.npy' % prompt_id
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
