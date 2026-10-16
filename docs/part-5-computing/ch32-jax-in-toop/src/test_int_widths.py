import ast
import sys

import numpy as np
from int_widths import jax_report, jax_report_in_subprocess, numpy_report


def test_numpy_2_default_int_is_64_bit_but_long_follows_the_c_long():
    report = numpy_report()
    assert int(np.__version__.split('.')[0]) >= 2
    assert report['dtype(int)'] == 'int64' and report['intp'] == 'int64'
    assert report["dtype('long')"] == ('int32' if sys.platform == 'win32' else 'int64')
    assert report['randint(2**32) works']


def test_jax_int_follows_the_x64_flag():
    with_x64 = ast.literal_eval(jax_report_in_subprocess(True))
    without = ast.literal_eval(jax_report_in_subprocess(False))
    assert with_x64['jnp.full(3, 7, dtype=int).dtype'] == 'int64' and with_x64['int_max()'] == 2**63 - 1
    assert without['jnp.full(3, 7, dtype=int).dtype'] == 'int32' and without['int_max()'] == 2**31 - 1
    assert with_x64['int64 max stored as int32'] == -1


def test_in_process_report_is_consistent():
    report = jax_report()
    assert report['jnp.full(3, 7, dtype=int).dtype'] == ('int64' if report['x64'] else 'int32')
