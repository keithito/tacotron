from util.numbers import normalize


def test_normalize_numbers():
  assert normalize('1') == 'one'
  assert normalize('15') == 'fifteen'
  assert normalize('24') == 'twenty-four'
  assert normalize('100') == 'one hundred'
  assert normalize('101') == 'one hundred one'
  assert normalize('456') == 'four hundred fifty-six'
  assert normalize('1000') == 'one thousand'
  assert normalize('1800') == 'eighteen hundred'
  assert normalize('2,000') == 'two thousand'
  assert normalize('3000') == 'three thousand'
  assert normalize('18000') == 'eighteen thousand'
  assert normalize('24,000') == 'twenty-four thousand'
  assert normalize('124,001') == 'one hundred twenty-four thousand one'
  assert normalize('6.4 sec') == 'six point four sec'


def test_normalize_ordinals():
  assert normalize('1st') == 'first'
  assert normalize('2nd') == 'second'
  assert normalize('9th') == 'ninth'
  assert normalize('243rd place') == 'two hundred and forty-third place'


def test_normalize_dates():
  assert normalize('1400') == 'fourteen hundred'
  assert normalize('1901') == 'nineteen oh one'
  assert normalize('1999') == 'nineteen ninety-nine'
  assert normalize('2000') == 'two thousand'
  assert normalize('2004') == 'two thousand four'
  assert normalize('2010') == 'twenty ten'
  assert normalize('2012') == 'twenty twelve'
  assert normalize('2025') == 'twenty twenty-five'
  assert normalize('September 11, 2001') == 'September eleven, two thousand one'
  assert normalize('July 26, 1984.') == 'July twenty-six, nineteen eighty-four.'


def test_normalize_money():
  assert normalize('$0.00') == 'zero dollars'
  assert normalize('$1') == 'one dollar'
  assert normalize('$10') == 'ten dollars'
  assert normalize('$.01') == 'one cent'
  assert normalize('$0.25') == 'twenty-five cents'
  assert normalize('$5.00') == 'five dollars'
  assert normalize('$5.01') == 'five dollars, one cent'
  assert normalize('$135.99.') == 'one hundred thirty-five dollars, ninety-nine cents.'
  assert normalize('$40,000') == 'forty thousand dollars'
  assert normalize('for £2500!') == 'for twenty-five hundred pounds!'
