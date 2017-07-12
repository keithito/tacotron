# tacotron

An implementation of Google's Tacotron speech synthesis model in Tensorflow.


### Audio Samples

  * Listen to **[Audio Samples](https://keithito.github.io/audio-samples/)** after training for 390k steps (~5 days).
  * The model is still training. I'll update the samples if the quality gets better.


## Background

Earlier this year, Google published a paper, [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. This is an
attempt to provide an open-source implementation of the model described in their paper.

The quality isn't as good as Google's demo yet, but hopefully it will get there someday :-).



## Quick Start

### Installing dependencies
```
pip install -r requirements.txt
```


### Using a pre-trained model

1. **Download and unpack a model**:
   ```
   curl http://data.keithito.com/data/speech/tacotron-20170712.tar.bz2 | tar x -C /tmp
   ```

2. **Run the demo server**:
   ```
   python3 demo_server.py --checkpoint /tmp/tacotron-20170712/model.ckpt
   ```

3. **Point your browser at [localhost:9000](http://localhost:9000)**
   * Type what you want to synthesize



### Training

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

   You can use other datasets if you convert them to the right format. See
   [ljspeech.py](datasets/ljspeech.py) for an example.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for LJ Speech:
   ```
   tacotron
     |- LJSpeech-1.0
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

5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   ```
   python demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to [localhost:9000](http://localhost:9000) and type what you want to speak.


## Miscellaneous Notes

  * [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) seems to improve
    training speed and avoids occasional slowdowns seen with the default allocator. You
    can enable it by installing it and setting `LD_PRELOAD=/usr/lib/libtcmalloc.so`.

  * You can train with [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) by downloading the
    dictionary to ~/tacotron/training and then passing the flag `--hparams="use_cmudict=True"` to
    train.py. This will allow you to pass ARPAbet phonemes enclosed in curly braces at eval
    time to force a particular pronunciation, e.g. `Turn left on {HH AW1 S S T AH0 N} Street.`

  * If you pass a Slack incoming webhook URL as the `--slack_url` flag to train.py, it will send
    you progress updates every 1000 steps.


## Other Implementations
  * By Alex Barron: https://github.com/barronalex/Tacotron
  * By Kyubyong Park: https://github.com/Kyubyong/tacotron
