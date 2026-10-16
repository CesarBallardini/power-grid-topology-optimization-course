"""Resistance and Joule losses of a transmission conductor, and the energy stored in L and C (Chapter 7 exercises).

Exercise 1: R = ρ L / A and P = R I². Exercise 2: a two-conductor line carrying P at voltage V has I = P / V and
losses 2 R I², which scale as 1 / V². Exercise 4: W_L = ½ L I², W_C = ½ C V².
"""

RHO_AL = 2.82e-8  # Ω·m, aluminium
LENGTH = 100e3  # m
AREA = 500e-6  # m²


# --8<-- [start:losses]
def resistance(rho: float = RHO_AL, length: float = LENGTH, area: float = AREA) -> float:
    """R = ρ L / A in ohms."""
    return rho * length / area


def joule_losses(current: float, r: float | None = None) -> float:
    """P = R I² in watts for one conductor (``r`` defaults to the Exercise 1 conductor)."""
    r = resistance() if r is None else r
    return r * current**2


def line_losses(power: float, voltage: float, conductors: int = 2, r: float | None = None) -> tuple[float, float]:
    """Current I = P / V and the losses of ``conductors`` conductors each carrying it, in (A, W)."""
    current = power / voltage
    return current, conductors * joule_losses(current, r)


# --8<-- [end:losses]


def inductor_energy(inductance: float, current: float) -> float:
    """W = ½ L I² in joules."""
    return 0.5 * inductance * current**2


def capacitor_energy(capacitance: float, voltage: float) -> float:
    """W = ½ C V² in joules."""
    return 0.5 * capacitance * voltage**2


def main() -> None:
    """Print the numbers quoted in the solutions of Exercises 1, 2 and 4."""
    r = resistance()
    print(f'Exercise 1: R = {r:.3f} ohm, losses at 1000 A = {joule_losses(1000.0) / 1e6:.3f} MW')
    for v in (200e3, 400e3):
        i, p = line_losses(200e6, v)
        pct = p / 200e6 * 100
        print(f'Exercise 2: {v / 1e3:.0f} kV -> I = {i:.0f} A, losses = {p / 1e6:.2f} MW ({pct:.2f} % of 200 MW)')
    _, p200 = line_losses(200e6, 200e3)
    _, p400 = line_losses(200e6, 400e3)
    print(f'            ratio of losses = {p200 / p400:.1f} = (400/200)^2')
    print(
        f'Exercise 4: inductor 1 H at 1000 A: {inductor_energy(1.0, 1000.0) / 1e3:.0f} kJ; '
        f'capacitor 1 uF at 230.9 kV: {capacitor_energy(1e-6, 230.9e3) / 1e3:.2f} kJ'
    )


if __name__ == '__main__':
    main()
