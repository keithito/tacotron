# Training Data


This repo supports the following speech datasets:
  * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
  * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

You can use any other dataset if you write a preprocessor for it.


### Writing a Preprocessor

Each training example consists of:
  1. The text that was spoken
  2. A mel-scale spectrogram of the audio
  3. A linear-scale spectrogram of the audio

The preprocessor is responsible for generating these. See [ljspeech.py](datasets/ljspeech.py) for a
heavily-commented example.

For each training example, a preprocessor should:

  1. Load the audio file:
     ```python
     wav = audio.load_wav(wav_path)
     ```

  2. Compute linear-scale and mel-scale spectrograms (float32 numpy arrays):
     ```python
     spectrogram = audio.spectrogram(wav).astype(np.float32)
     mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
     ```

  3. Save the spectrograms to disk:
     ```python
     np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
     np.save(os.path.join(out_dir, mel_spectrogram_filename), mel_spectrogram.T,  allow_pickle=False)
     ```
     Note that the transpose of the matrix returned by `audio.spectrogram` is saved so that it's
     in time-major format.

  4. Generate a tuple `(spectrogram_filename, mel_spectrogram_filename, n_frames, text)` to
     write to train.txt. n_frames is just the length of the time axis of the spectrogram.


After you've written your preprocessor, you can add it to [preprocess.py](preprocess.py) by
following the example of the other preprocessors in that file.



### Text Processing During Training and Eval

Some additional processing is done to the text during training and eval. The text is run
through the `to_sequence` function in [textinput.py](util/textinput.py).

This performs several transformations:
  1. Leading and trailing whitespace and quotation marks are removed.
  2. Text is converted to ASCII by removing diacritics (e.g. "Crème brûlée" becomes "Creme brulee").
  3. Numbers are converted to strings using the heuristics in [numbers.py](util/numbers.py).
     *This is specific to English*.
  4. Abbreviations are expanded (e.g. "Mr" becomes "Mister"). *This is also specific to English*.
  5. Characters outside the input alphabet (ASCII characters and some punctuation) are removed.
  6. Whitespace is collapsed so that every sequence of whitespace becomes a single ASCII space.

**Several of these steps are inappropriate for non-English text and you may want to disable or
modify them if you are not using English training data.**
