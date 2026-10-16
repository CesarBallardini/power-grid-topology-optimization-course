import math

import floating_point as fp
import numpy as np


def test_finfo_table():
    info = fp.finfo_table()
    assert info['float32']['eps'] == 2.0**-23 and info['float64']['eps'] == 2.0**-52
    assert info['float32']['decimal_digits'] == 6 and info['float64']['decimal_digits'] == 15
    assert info['float32']['mantissa_bits'] == 23 and info['float64']['mantissa_bits'] == 52


def test_the_three_surprises():
    s = fp.surprises()
    assert s['0.1 + 0.2 == 0.3'] is False
    assert s['1e16 + 1 - 1e16'] == 0.0  # 1 is below half an ulp of 1e16 (ulp = 2)
    assert math.isnan(s['np.sum([1.0, np.nan])'])
    assert 1e15 + 1 - 1e15 == 1.0  # one decade lower the ulp is 0.125 and the 1 survives


def test_nan_propagates_through_every_reduction_and_product():
    out = fp.nan_spreads(np.array([1.0, np.nan, 3.0]))
    assert all(math.isnan(out[k]) for k in ('sum', 'mean', 'max'))
    assert out['matrix_product_entries_nan'] == 3  # one NaN input poisons every output of a dense product
