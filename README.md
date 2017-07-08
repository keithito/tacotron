# tacotron

An implementation of Google's Tacotron speech synthesis model in Tensorflow.


## Overview

Earlier this year, Google published a paper, [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/pdf/1703.10135.pdf),
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs.

Google [released](https://google.github.io/tacotron) some nice audio samples that their model
generated but didn't provide their source code or training data. This is an attempt to
implement the model described in their paper.

Output after training for 185K steps (~2 days):

  * [Audio Samples](https://keithito.github.io/audio-samples/)

The quality isn't as good as what Google demoed. But hopefully it will get there someday :-).



## Quick Start

### Installing dependencies
```
pip install -r requirements.txt
```


### Using a pre-trained model

1. Download and unpack a model:
   ```
   curl http://data.keithito.com/data/speech/tacotron-20170708.tar.bz2 | tar x -C /tmp
   ```

2. Run the demo server:
   ```
   python3 demo_server.py --checkpoint /tmp/tacotron-20170708/model.ckpt
   ```

3. Point your browser at [localhost:9000](http://localhost:9000) and type!



### Training

1. Download a speech dataset. The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset) (Public Domain)
    * [Blizzard 2012](http://www.cstr.ed.ac.uk/projects/blizzard/2012/phase_one) (Creative Commons Attribution Share-Alike)

   You can use other datasets if you convert them to the right format. See
   [ljspeech.py](datasets/ljspeech.py) for an example.


2. Unpack the dataset into `~/tacotron`. After unpacking, your tree should look like this for
   LJ Speech:
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

3. Preprocess the data
   ```
   python3 preprocess.py --dataset ljspeech
   ```
   *Use --dataset blizzard for Blizzard data*

4. Train
   ```
   python3 train.py
   ```
   *Note: using [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) seems to
   improve training performance.*

5. Monitor with Tensorboard (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`. You can also pass a Slack webhook URL as the `--slack_url`
   flag, and it will send you progress updates.



## Other Implementations

  * Alex Barron has some nice results from his implementation trained on the
    [Nancy Corpus](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011):
    https://github.com/barronalex/Tacotron

  * Kyubyong Park has a very promising implementation trained on the World English Bible here:
    https://github.com/Kyubyong/tacotron
