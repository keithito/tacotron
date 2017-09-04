from text.numbers import normalize_numbers


def test_normalize_numbers():
  assert normalize_numbers('1') == 'one'
  assert normalize_numbers('15') == 'fifteen'
  assert normalize_numbers('24') == 'twenty-four'
  assert normalize_numbers('100') == 'one hundred'
  assert normalize_numbers('101') == 'one hundred one'
  assert normalize_numbers('456') == 'four hundred fifty-six'
  assert normalize_numbers('1000') == 'one thousand'
  assert normalize_numbers('1800') == 'eighteen hundred'
  assert normalize_numbers('2,000') == 'two thousand'
  assert normalize_numbers('3000') == 'three thousand'
  assert normalize_numbers('18000') == 'eighteen thousand'
  assert normalize_numbers('24,000') == 'twenty-four thousand'
  assert normalize_numbers('124,001') == 'one hundred twenty-four thousand one'
  assert normalize_numbers('6.4 sec') == 'six point four sec'


def test_normalize_ordinals():
  assert normalize_numbers('1st') == 'first'
  assert normalize_numbers('2nd') == 'second'
  assert normalize_numbers('9th') == 'ninth'
  assert normalize_numbers('243rd place') == 'two hundred and forty-third place'


def test_normalize_dates():
  assert normalize_numbers('1400') == 'fourteen hundred'
  assert normalize_numbers('1901') == 'nineteen oh one'
  assert normalize_numbers('1999') == 'nineteen ninety-nine'
  assert normalize_numbers('2000') == 'two thousand'
  assert normalize_numbers('2004') == 'two thousand four'
  assert normalize_numbers('2010') == 'twenty ten'
  assert normalize_numbers('2012') == 'twenty twelve'
  assert normalize_numbers('2025') == 'twenty twenty-five'
  assert normalize_numbers('September 11, 2001') == 'September eleven, two thousand one'
  assert normalize_numbers('July 26, 1984.') == 'July twenty-six, nineteen eighty-four.'


def test_normalize_money():
  assert normalize_numbers('$0.00') == 'zero dollars'
  assert normalize_numbers('$1') == 'one dollar'
  assert normalize_numbers('$10') == 'ten dollars'
  assert normalize_numbers('$.01') == 'one cent'
  assert normalize_numbers('$0.25') == 'twenty-five cents'
  assert normalize_numbers('$5.00') == 'five dollars'
  assert normalize_numbers('$5.01') == 'five dollars, one cent'
  assert normalize_numbers('$135.99.') == 'one hundred thirty-five dollars, ninety-nine cents.'
  assert normalize_numbers('$40,000') == 'forty thousand dollars'
  assert normalize_numbers('for £2500!') == 'for twenty-five hundred pounds!'
