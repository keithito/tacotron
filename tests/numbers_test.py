from text.numbers import normalize_numbers


def test_normalize_numbers():
  assert normalize_numbers('0') == 'zero'
  assert normalize_numbers('1') == 'one'
  assert normalize_numbers('15') == 'fifteen'
  assert normalize_numbers('24') == 'twenty four'
  assert normalize_numbers('100') == 'one hundred'
  assert normalize_numbers('101') == 'one hundred one'
  assert normalize_numbers('456') == 'four hundred fifty six'
  assert normalize_numbers('1000') == 'one thousand'
  assert normalize_numbers('1800') == 'eighteen hundred'
  assert normalize_numbers('2,000') == 'two thousand'
  assert normalize_numbers('3000') == 'three thousand'
  assert normalize_numbers('18000') == 'eighteen thousand'
  assert normalize_numbers('24,000') == 'twenty four thousand'
  assert normalize_numbers('124,001') == 'one hundred twenty four thousand one'
  assert normalize_numbers('999,999') == 'nine hundred ninety nine thousand nine hundred ninety nine'
  assert normalize_numbers('1000000002') == 'one billion two'
  assert normalize_numbers('1200000000') == 'one billion two hundred million'
  assert normalize_numbers('19800000004001') == 'nineteen trillion eight hundred billion four thousand one'
  assert normalize_numbers('712000000000000000') == 'seven hundred twelve quadrillion'
  assert normalize_numbers('1000000000000000000') == '1000000000000000000'
  assert normalize_numbers('6.4 sec') == 'six point four sec'


def test_normalize_ordinals():
  assert normalize_numbers('1st') == 'first'
  assert normalize_numbers('2nd') == 'second'
  assert normalize_numbers('5th') == 'fifth'
  assert normalize_numbers('9th') == 'ninth'
  assert normalize_numbers('15th') == 'fifteenth'
  assert normalize_numbers('212th street') == 'two hundred twelfth street'
  assert normalize_numbers('243rd place') == 'two hundred forty third place'
  assert normalize_numbers('1025th') == 'one thousand twenty fifth'
  assert normalize_numbers('1000000th') == 'one millionth'


def test_normalize_money():
  assert normalize_numbers('$0.00') == 'zero dollars'
  assert normalize_numbers('$1') == 'one dollar'
  assert normalize_numbers('$10') == 'ten dollars'
  assert normalize_numbers('$.01') == 'one cent'
  assert normalize_numbers('$0.25') == 'twenty five cents'
  assert normalize_numbers('$5.00') == 'five dollars'
  assert normalize_numbers('$5.01') == 'five dollars, one cent'
  assert normalize_numbers('$135.99.') == 'one hundred thirty five dollars, ninety nine cents.'
  assert normalize_numbers('$40,000') == 'forty thousand dollars'
  assert normalize_numbers('for £2500!') == 'for twenty five hundred pounds!'
