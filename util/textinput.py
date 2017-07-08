import re
import unicodedata
from util import cmudict, numbers


# Input alphabet (63 symbols), plus ARPAbet (84 symbols):
_pad         = '_'
_eos         = '~'
_uppercase   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_lowercase   = 'abcdefghijklmnopqrstuvwxyz'
_punctuation = '!\'(),-.:;?'
_space       = ' '

_valid_input_chars = _uppercase + _lowercase + _punctuation + _space
_trans_table = str.maketrans({chr(i): ' ' for i in range(256) if chr(i) not in _valid_input_chars})

_normal_symbols = _pad + _eos + _valid_input_chars
_num_normal_symbols = len(_normal_symbols)
_char_to_id = {c: i for i, c in enumerate(_normal_symbols)}
_id_to_char = {i: c for i, c in enumerate(_normal_symbols)}
_arpabet_to_id = {sym: i + _num_normal_symbols for i, sym in enumerate(cmudict.valid_symbols)}
_id_to_arpabet = {i + _num_normal_symbols: sym for i, sym in enumerate(cmudict.valid_symbols)}
_arpabet_re = re.compile(r'(.*?)\{([A-Z0-2 ]+?)\}(.*)')
_num_symbols = _num_normal_symbols + len(cmudict.valid_symbols)
_whitespace_re = re.compile(r'\s+')


def num_symbols():
  '''Returns number of symbols in the alphabet.'''
  return _num_symbols


def to_sequence(text, force_lowercase=True, expand_abbreviations=True):
  '''Converts a string of text to a sequence of IDs for the symbols in the text'''
  text = text.strip()
  text = text.replace('"', '')
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()

  sequence = []
  while len(text):
    m = _arpabet_re.match(text)
    if not m:
      sequence += _text_to_sequence(text, force_lowercase, expand_abbreviations)
      break
    sequence += _text_to_sequence(m.group(1), force_lowercase, expand_abbreviations)
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)
  sequence.append(_char_to_id[_eos])
  return sequence


def to_string(sequence, remove_eos=False):
  '''Returns the string for a sequence of characters.'''
  s = ''
  for sym in sequence:
    if sym < _num_normal_symbols:
      s += _id_to_char[sym]
    elif sym < _num_symbols:
      s += '{%s}' % _id_to_arpabet[sym]
  s = s.replace('}{', ' ')
  if remove_eos and s[-1] == _eos:
    s = s[:-1]
  return s


def _text_to_sequence(text, force_lowercase, expand_abbreviations):
  text = numbers.normalize(text)
  text = text.translate(_trans_table)
  if force_lowercase:
    text = text.lower()
  if expand_abbreviations:
    text = _expand_abbreviations(text)
  text = re.sub(_whitespace_re, ' ', text)
  return [_char_to_id[c] for c in text]


_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def _expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def _arpabet_to_sequence(text):
  return [_arpabet_to_id[s] for s in text.split() if s in _arpabet_to_id]
