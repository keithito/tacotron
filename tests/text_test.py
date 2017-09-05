from text import cleaners, symbols, text_to_sequence, sequence_to_text
from unidecode import unidecode


def test_symbols():
  assert len(symbols) >= 3
  assert symbols[0] == '_'
  assert symbols[1] == '~'


def test_text_to_sequence():
  assert text_to_sequence('', []) == [1]
  assert text_to_sequence('Hi!', []) == [9, 36, 54, 1]
  assert text_to_sequence('"A"_B', []) == [2, 3, 1]
  assert text_to_sequence('A {AW1 S} B', []) == [2, 64, 83, 132, 64, 3, 1]
  assert text_to_sequence('Hi', ['lowercase']) == [35, 36, 1]
  assert text_to_sequence('A {AW1 S}  B', ['english_cleaners']) == [28, 64, 83, 132, 64, 29, 1]


def test_sequence_to_text():
  assert sequence_to_text([]) == ''
  assert sequence_to_text([1]) == '~'
  assert sequence_to_text([9, 36, 54, 1]) == 'Hi!~'
  assert sequence_to_text([2, 64, 83, 132, 64, 3]) == 'A {AW1 S} B'


def test_collapse_whitespace():
  assert cleaners.collapse_whitespace('') == ''
  assert cleaners.collapse_whitespace('  ') == ' '
  assert cleaners.collapse_whitespace('x') == 'x'
  assert cleaners.collapse_whitespace(' x.  y,  \tz') == ' x. y, z'


def test_convert_to_ascii():
  assert cleaners.convert_to_ascii("raison d'être") == "raison d'etre"
  assert cleaners.convert_to_ascii('grüß gott') == 'gruss gott'
  assert cleaners.convert_to_ascii('안녕') == 'annyeong'
  assert cleaners.convert_to_ascii('Здравствуйте') == 'Zdravstvuite'


def test_lowercase():
  assert cleaners.lowercase('Happy Birthday!') == 'happy birthday!'
  assert cleaners.lowercase('CAFÉ') == 'café'


def test_expand_abbreviations():
  assert cleaners.expand_abbreviations('mr. and mrs. smith') == 'mister and misess smith'


def test_expand_numbers():
  assert cleaners.expand_numbers('3 apples and 44 pears') == 'three apples and forty-four pears'
  assert cleaners.expand_numbers('$3.50 for gas.') == 'three dollars, fifty cents for gas.'


def test_cleaner_pipelines():
  text = 'Mr. Müller ate  2 Apples'
  assert cleaners.english_cleaners(text) == 'mister muller ate two apples'
  assert cleaners.transliteration_cleaners(text) == 'mr. muller ate 2 apples'
  assert cleaners.basic_cleaners(text) == 'mr. müller ate 2 apples'

