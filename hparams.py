import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
  # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
  cleaners='english_cleaners',

  # Audio:
  num_mels=80,
  num_freq=1025,
  min_mel_freq=125,
  max_mel_freq=7600,
  sample_rate=20000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  min_level_db=-100,
  ref_level_db=20,

  # Model:
  outputs_per_step=5,

  embedding_dim=512,

  encoder_conv_layers=3,
  encoder_conv_width=5,
  encoder_conv_channels=512,
  encoder_lstm_units=256,     # For each direction

  attention_depth=128,

  decoder_lstm_units=1024,

  postnet_conv_layers=5,
  postnet_conv_width=5,
  postnet_conv_channels=512,

  expand_conv_layers=5,
  expand_conv_width=5,
  expand_conv_channels=512,
  expand_lstm_units=256,     # For each direction


  # Training:
  batch_size=32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.001,
  learning_rate_decay_halflife=100000,
  use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

  # Eval:
  max_iters=200,
  griffin_lim_iters=60,
  power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
