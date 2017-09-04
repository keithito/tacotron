'''
Defines the set of symbols used in text input to the model.

The default works well for English. For non-English datasets, update _characters to be the set of
characters in the dataset. The "cleaners" hyperparameter should also be changed to be
"basic_pipeline" or a custom set of steps for the dataset (see cleaners.py for more info).
'''
from text import cmudict

_pad        = '_'
_eos        = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet
