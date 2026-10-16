"""The branch list of the Chapter 2 lab as a pandas table, through a CSV file and back (Exercise 6).

``python branch_table.py`` writes the table to an in-memory CSV, rebuilds the incidence matrix from it, adds the
branch currents of Lab step 3 and prints the table sorted by absolute current.
"""

import io

import numpy as np
import pandas as pd
from lab_network import fr, g, to
from laplacian import incidence
from resistor_lab import INJECTION, N_NODES, solve


def branch_frame() -> pd.DataFrame:
    """The lab's branch list with columns ``from``, ``to`` and ``g``."""
    return pd.DataFrame({'from': fr, 'to': to, 'g': g})


def round_trip(frame: pd.DataFrame, path_or_buffer) -> pd.DataFrame:
    """Write ``frame`` as CSV without the index and read it back; ``path_or_buffer`` is a path or a text buffer."""
    frame.to_csv(path_or_buffer, index=False)
    if isinstance(path_or_buffer, io.StringIO):
        path_or_buffer.seek(0)
    return pd.read_csv(path_or_buffer)


def incidence_from_frame(frame: pd.DataFrame, n: int) -> np.ndarray:
    """The incidence matrix rebuilt from the table's ``from`` and ``to`` columns."""
    return incidence(frame['from'].to_numpy(), frame['to'].to_numpy(), n)


def with_currents(frame: pd.DataFrame, i: np.ndarray = INJECTION, ground: int = 0) -> pd.DataFrame:
    """The table with a ``current`` column from the nodal solve of Lab step 3, sorted by absolute current."""
    a = incidence_from_frame(frame, N_NODES)
    _, cur, _ = solve(a, frame['g'].to_numpy(dtype=float), i, ground)
    out = frame.assign(current=cur)
    return out.iloc[np.argsort(-np.abs(cur), kind='stable')].reset_index(drop=True)


def main() -> None:
    """Round-trip the table through CSV, rebuild A, add the currents and print the sorted table."""
    frame = round_trip(branch_frame(), io.StringIO())
    a = incidence_from_frame(frame, N_NODES)
    same = np.array_equal(a, incidence(np.array(fr), np.array(to), N_NODES))
    print('A rebuilt from the CSV equals the original:', same)
    print(with_currents(frame).to_string(index=False))


if __name__ == '__main__':
    main()
