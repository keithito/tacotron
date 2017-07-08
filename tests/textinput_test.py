from util.textinput import num_symbols, to_sequence, to_string


def text_num_symbols():
  assert num_symbols() == 147


def test_to_sequence():
  assert to_sequence('') == [1]
  assert to_sequence('H', force_lowercase=False) == [9, 1]
  assert to_sequence('H', force_lowercase=True) == [35, 1]
  assert to_sequence('Hi.', force_lowercase=False) == [9, 36, 60, 1]


def test_whitespace_nomalization():
  assert round_trip('') == '~'
  assert round_trip(' ') == '~'
  assert round_trip('x') == 'x~'
  assert round_trip(' x ') == 'x~'
  assert round_trip(' x.  y,z ') == 'x. y,z~'
  assert round_trip('X:  Y') == 'X: Y~'


def test_valid_chars():
  assert round_trip('x') == 'x~'
  assert round_trip('Hello') == 'Hello~'
  assert round_trip('3 apples and 44 bananas') == 'three apples and forty-four bananas~'
  assert round_trip('$3.50 for gas.') == 'three dollars, fifty cents for gas.~'
  assert round_trip('Hello, world!') == 'Hello, world!~'
  assert round_trip("What (time-out)! He\'s going where?") == "What (time-out)! He\'s going where?~"


def test_invalid_chars():
  assert round_trip('^') == ' ~'
  assert round_trip('A~^B') == 'A B~'
  assert round_trip('"Finally," she said, "it ended."') == 'Finally, she said, it ended.~'


def test_unicode():
  assert round_trip('naïve café') == 'naive cafe~'
  assert round_trip("raison d'être") == "raison d'etre~"


def test_arpabet():
  assert to_sequence('{AE0 D}') == [70, 91, 1]
  assert round_trip('{AE0 D V ER1 S}') == '{AE0 D V ER1 S}~'
  assert round_trip('{AE0 D V ER1 S} circumstances') == '{AE0 D V ER1 S} circumstances~'
  assert round_trip('In {AE0 D V ER1 S} circumstances') == 'In {AE0 D V ER1 S} circumstances~'
  assert round_trip('{AE0 D V ER1 S} {AE0 D S}') == '{AE0 D V ER1 S} {AE0 D S}~'
  assert round_trip('X {AE0 D} Y  {AE0 D} Z') == 'X {AE0 D} Y {AE0 D} Z~'


def test_abbreviations():
  assert round_trip('mr. rogers and dr. smith.') == 'mister rogers and doctor smith.~'
  assert round_trip('hit it with a hammr.') == 'hit it with a hammr.~'


def round_trip(x):
  return to_string(to_sequence(x, force_lowercase=False, expand_abbreviations=True))
