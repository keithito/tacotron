# Tacotron

An implementation of Tacotron speech synthesis in TensorFlow.


### Audio Samples

  * **[Audio Samples](https://keithito.github.io/audio-samples/)** from models trained using this repo.
    * The first set was trained for 441K steps on the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
      * Speech started to become intelligible around 20K steps.
    * The second set was trained by [@MXGray](https://github.com/MXGray) for 140K steps on the [Nancy Corpus](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/).


### Recent Updates

1. @npuichigo [fixed](https://github.com/keithito/tacotron/pull/205) a bug where dropout was not being applied in the prenet.

2. @begeekmyfriend created a [fork](https://github.com/begeekmyfriend/tacotron) that adds location-sensitive attention and the stop token from the [Tacotron 2](https://arxiv.org/abs/1712.05884) paper. This can greatly reduce the amount of data required to train a model.


## Background

In April 2017, Google published a paper, [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. This is an
independent attempt to provide an open-source implementation of the model described in their paper.

The quality isn't as good as Google's demo yet, but hopefully it will get there someday :-).
Pull requests are welcome!



## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model

1. **Download and unpack a model**:
   ```
   curl https://data.keithito.com/data/speech/tacotron-20180906.tar.gz | tar xzC /tmp
   ```

2. **Run the demo server**:
   ```
   python3 demo_server.py --checkpoint /tmp/tacotron-20180906/model.ckpt
   ```

3. **Point your browser at localhost:9000**
   * Type what you want to synthesize



### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

   You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](TRAINING_DATA.md) for more info.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for LJ Speech:
   ```
   tacotron
     |- LJSpeech-1.1
         |- metadata.csv
         |- wavs
   ```

   or like this for Blizzard 2012:
   ```
   tacotron
     |- Blizzard2012
         |- ATrampAbroad
         |   |- sentence_index.txt
         |   |- lab
         |   |- wav
         |- TheManThatCorruptedHadleyburg
             |- sentence_index.txt
             |- lab
             |- wav
   ```

3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset ljspeech
   ```
     * Use `--dataset blizzard` for Blizzard data

4. **Train a model**
   ```
   python3 train.py
   ```

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.
   The default hyperparameters are recommended for LJ Speech and other English-language data.
   See [TRAINING_DATA.md](TRAINING_DATA.md) for other languages.


5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.


## Notes and Common Issues

  * [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) seems to improve
    training speed and avoids occasional slowdowns seen with the default allocator. You
    can enable it by installing it and setting `LD_PRELOAD=/usr/lib/libtcmalloc.so`. With TCMalloc,
    you can get around 1.1 sec/step on a GTX 1080Ti.

  * You can train with [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) by downloading the
    dictionary to ~/tacotron/training and then passing the flag `--hparams="use_cmudict=True"` to
    train.py. This will allow you to pass ARPAbet phonemes enclosed in curly braces at eval
    time to force a particular pronunciation, e.g. `Turn left on {HH AW1 S S T AH0 N} Street.`

  * If you pass a Slack incoming webhook URL as the `--slack_url` flag to train.py, it will send
    you progress updates every 1000 steps.

  * Occasionally, you may see a spike in loss and the model will forget how to attend (the
    alignments will no longer make sense). Although it will recover eventually, it may
    save time to restart at a checkpoint prior to the spike by passing the
    `--restore_step=150000` flag to train.py (replacing 150000 with a step number prior to the
    spike). **Update**: a recent [fix](https://github.com/keithito/tacotron/pull/7) to gradient
    clipping by @candlewill may have fixed this.
    
  * During eval and training, audio length is limited to `max_iters * outputs_per_step * frame_shift_ms`
    milliseconds. With the defaults (max_iters=200, outputs_per_step=5, frame_shift_ms=12.5), this is
    12.5 seconds.
    
    If your training examples are longer, you will see an error like this:
    `Incompatible shapes: [32,1340,80] vs. [32,1000,80]`
    
    To fix this, you can set a larger value of `max_iters` by passing `--hparams="max_iters=300"` to
    train.py (replace "300" with a value based on how long your audio is and the formula above).
    
  * Here is the expected loss curve when training on LJ Speech with the default hyperparameters:
    ![Loss curve](https://user-images.githubusercontent.com/1945356/36077599-c0513e4a-0f21-11e8-8525-07347847720c.png)


## Other Implementations
  * By Alex Barron: https://github.com/barronalex/Tacotron
  * By Kyubyong Park: https://github.com/Kyubyong/tacotron
