"""Floating-point facts behind ToOp's float64 policy and NaN assertion (Chapter 4, Exercise 4)."""

import numpy as np


def finfo_table() -> dict[str, dict[str, float | int]]:
    """Machine epsilon, smallest normal, largest value and decimal digits of float32 and float64."""
    out = {}
    for dtype in (np.float32, np.float64):
        info = np.finfo(dtype)
        out[np.dtype(dtype).name] = {
            'eps': float(info.eps),
            'tiny': float(info.tiny),
            'max': float(info.max),
            'decimal_digits': int(info.precision),
            'mantissa_bits': int(info.nmant),
        }
    return out


def surprises() -> dict[str, object]:
    """The three expressions of Exercise 4 and what they evaluate to."""
    with np.errstate(invalid='ignore'):
        return {
            '0.1 + 0.2 == 0.3': 0.1 + 0.2 == 0.3,
            '0.1 + 0.2': 0.1 + 0.2,
            '1e16 + 1 - 1e16': 1e16 + 1 - 1e16,
            'np.sum([1.0, np.nan])': float(np.sum([1.0, np.nan])),
            'np.float32(1e16) + 1 - np.float32(1e16)': float(np.float32(1e16) + np.float32(1) - np.float32(1e16)),
        }


def nan_spreads(values: np.ndarray) -> dict[str, float]:
    """How one NaN entry contaminates a sum, a mean, a max and a matrix product."""
    matrix = np.eye(len(values))
    return {
        'sum': float(np.sum(values)),
        'mean': float(np.mean(values)),
        'max': float(np.max(values)),
        'matrix_product_entries_nan': int(np.isnan(matrix @ values).sum()),
    }


def main() -> None:
    """Print the finfo table, the three expressions and the NaN propagation of Exercise 4."""
    for name, info in finfo_table().items():
        print(name, info)
    for expr, value in surprises().items():
        print(f'{expr:45s} -> {value!r}')
    print('NaN propagation:', nan_spreads(np.array([1.0, np.nan, 3.0])))


if __name__ == '__main__':
    main()
