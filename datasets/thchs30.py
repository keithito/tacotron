from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the THCHS30 dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the THCHS30 dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  # male voice (do not use) A5 A8 A9 A33 A35 B6 B8 B21 B34 C6 C8 D8
  # too silent (do not use) A36 B33 C14 D32

  trn_files = []

  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A14_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A19_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A23_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A32_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'A34_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B15_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B31_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'B32_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C17_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C18_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C19_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C20_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C21_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C23_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C31_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'C32_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D6_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D21_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'data', 'D31_*.trn'))

  for trn in trn_files:
    with open(trn) as f:
      f.readline()
      pinyin = f.readline().strip('\n')
      wav_file = trn[:-4]
      task = partial(_process_utterance, out_dir, index, wav_file, pinyin)
      futures.append(executor.submit(task))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, pinyin):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    pinyin: The pinyin of Chinese spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'thchs30-spec-%05d.npy' % index
  mel_filename = 'thchs30-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, pinyin)
