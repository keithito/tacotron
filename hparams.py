import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  # Text:
  force_lowercase=True,
  expand_abbreviations=True,
  use_cmudict=False,

  # Audio:
  num_mels=80,
  num_freq=1025,
  sample_rate=20000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,

  # Model:
  # TODO: add more configurable hparams
  outputs_per_step=5,

  # Training:
  batch_size=32,
  adam_beta1=0.9,
  adam_beta2=0.999,
  initial_learning_rate=0.002,
  decay_learning_rate=True,

  # Eval:
  max_iters=200,
  griffin_lim_iters=60,
  power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
