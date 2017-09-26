from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import librosa
import numpy as np
import os
import re
from util import audio


_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  wav_paths = glob.glob('%s/wav48/p*/*.wav' % in_dir)
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  for wav_path in wav_paths:
    text_path = wav_path.replace('wav48', 'txt').replace('wav', 'txt')
    if os.path.isfile(text_path):
      with open(text_path, 'r') as f:
        text = f.read().strip()
      futures.append(executor.submit(partial(_process_utterance, out_dir, wav_path, text)))
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, wav_path, text):
  wav = _trim_wav(audio.load_wav(wav_path))
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  name = os.path.splitext(os.path.basename(wav_path))[0]
  speaker_id = _speaker_re.match(name).group(1)
  spectrogram_filename = 'vctk-linear-%s.npy' % name
  mel_filename = 'vctk-mel-%s.npy' % name
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)


def _trim_wav(wav):
  '''Trims silence from the ends of the wav'''
  splits = librosa.effects.split(wav, _threshold_db, frame_length=1024, hop_length=512)
  return wav[_find_start(splits):_find_end(splits, len(wav))]


def _find_start(splits):
  for split_start, split_end in splits:
    if split_end - split_start > _min_samples:
      return max(0, split_start - _min_samples)
  return 0


def _find_end(splits, num_samples):
  for split_start, split_end in reversed(splits):
    if split_end - split_start > _min_samples:
      return min(num_samples, split_end + _min_samples)
  return num_samples
