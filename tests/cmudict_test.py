import io
from text import cmudict


test_data = '''
;;; # CMUdict  --  Major Version: 0.07
)PAREN  P ER EH N
'TIS  T IH Z
ADVERSE  AE0 D V ER1 S
ADVERSE(1)  AE1 D V ER2 S
ADVERSE(2)  AE2 D V ER1 S
ADVERSELY  AE0 D V ER1 S L IY0
ADVERSITY  AE0 D V ER1 S IH0 T IY2
BARBERSHOP  B AA1 R B ER0 SH AA2 P
YOU'LL  Y UW1 L
'''


def test_cmudict():
  c = cmudict.CMUDict(io.StringIO(test_data))
  assert len(c) == 6
  assert len(cmudict.valid_symbols) == 84
  assert c.lookup('ADVERSITY') == ['AE0 D V ER1 S IH0 T IY2']
  assert c.lookup('BarberShop') == ['B AA1 R B ER0 SH AA2 P']
  assert c.lookup("You'll") == ['Y UW1 L']
  assert c.lookup("'tis") == ['T IH Z']
  assert c.lookup('adverse') == [
    'AE0 D V ER1 S',
    'AE1 D V ER2 S',
    'AE2 D V ER1 S',
  ]
  assert c.lookup('') == None
  assert c.lookup('foo') == None
  assert c.lookup(')paren') == None


def test_cmudict_no_keep_ambiguous():
  c = cmudict.CMUDict(io.StringIO(test_data), keep_ambiguous=False)
  assert len(c) == 5
  assert c.lookup('adversity') == ['AE0 D V ER1 S IH0 T IY2']
  assert c.lookup('adverse') == None
