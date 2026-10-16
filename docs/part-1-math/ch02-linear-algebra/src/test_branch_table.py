import io

import numpy as np
from branch_table import branch_frame, incidence_from_frame, round_trip, with_currents
from lab_network import fr, to
from laplacian import incidence


def test_csv_round_trip_keeps_the_table(tmp_path):
    frame = branch_frame()
    back_mem = round_trip(frame, io.StringIO())
    back_file = round_trip(frame, tmp_path / 'branches.csv')
    assert back_mem.equals(frame) and back_file.equals(frame)
    assert list(frame.columns) == ['from', 'to', 'g']


def test_incidence_rebuilt_from_the_file():
    frame = round_trip(branch_frame(), io.StringIO())
    np.testing.assert_array_equal(incidence_from_frame(frame, 6), incidence(np.array(fr), np.array(to), 6))


def test_currents_sorted_by_absolute_value():
    table = with_currents(branch_frame())
    assert list(table.columns) == ['from', 'to', 'g', 'current']
    assert (np.diff(np.abs(table['current'])) <= 1e-12).all()
    assert (table.iloc[0]['from'], table.iloc[0]['to']) == (4, 5) and table.iloc[0]['current'] == 1.0
