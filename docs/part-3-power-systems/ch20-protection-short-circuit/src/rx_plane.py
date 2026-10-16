"""Distance protection in the R-X plane (Chapter 20): ToOp's polygon, apparent impedances and quadrilateral zones.

Lab Part B builds ToOp's "danger" polygon from ``angle``, ``r_i``, ``r_v`` and ``x_v``, tests whether the apparent
impedance of a loaded line falls inside it (with ToOp's warning factor, which divides |R| and |X|), finds the MVA
at which a loading path enters the zone, and derives the validity condition r_v tan(angle) <= x_v. Exercise 5 adds
quadrilateral zones 1 and 2 of a 100 km line and the infeed effect on the apparent impedance. Pure NumPy: the
point-in-polygon and segment-intersection tests are written out, so no geometry library is needed.
"""

import numpy as np

Point = tuple[float, float]


def toop_polygon(angle_deg: float, r_i: float, r_v: float, x_v: float) -> list[Point]:
    """ToOp's ``_build_poly`` vertices: (0,0), (r_i,0), (r_i, r_i tan a), (r_v, r_v tan a), (r_v, x_v), (0, x_v)."""
    t = np.tan(np.radians(angle_deg))
    return [(0.0, 0.0), (r_i, 0.0), (r_i, r_i * t), (r_v, r_v * t), (r_v, x_v), (0.0, x_v)]


def is_valid(angle_deg: float, r_i: float, r_v: float, x_v: float) -> bool:
    """The polygon is simple (no self-intersection) iff r_v tan(angle) <= x_v, with 0 <= r_i <= r_v."""
    return 0.0 <= r_i <= r_v and r_v * np.tan(np.radians(angle_deg)) <= x_v


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Proper intersection of the open segments p1p2 and q1q2 (shared endpoints do not count)."""

    def orient(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def is_simple(poly: list[Point]) -> bool:
    """True when no two non-adjacent edges cross (what shapely calls a valid polygon)."""
    n = len(poly)
    edges = [(poly[i], poly[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through the closing edge
            if _segments_cross(*edges[i], *edges[j]):
                return False
    return True


def contains(poly: list[Point], point: Point) -> bool:
    """Ray casting: is the point inside the polygon (boundary counts as inside)?"""
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (
            min(y1, y2) <= y <= max(y1, y2)
            and min(x1, x2) <= x <= max(x1, x2)
            and abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-12
        ):
            return True  # on an edge
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def apparent_impedance(v_ll_kv: float, p_mw: float, q_mvar: float) -> Point:
    """(R, X) in ohm seen by a relay: |Z| = V_LL^2 / |S| at the angle of P + jQ (ToOp's ``get_complex_impedance``)."""
    s = complex(p_mw, q_mvar)
    if s == 0:
        return (np.inf, np.inf)
    z = v_ll_kv**2 / abs(s) * s / abs(s)
    return (z.real, z.imag)


def trips(poly: list[Point], z: Point, factor: float = 1.0) -> bool:
    """ToOp's test: (|R|, |X|) divided by the zone factor inside the polygon (so the test is non-directional)."""
    return contains(poly, (abs(z[0]) / factor, abs(z[1]) / factor))


def boundary_distance(poly: list[Point], phi_rad: float) -> float:
    """Largest |Z| on the ray from the origin at angle phi that is still inside the polygon (ray-edge intersection)."""
    d = np.array([np.cos(phi_rad), np.sin(phi_rad)])
    best = 0.0
    n = len(poly)
    for i in range(n):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % n])
        e = b - a
        det = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(det) < 1e-12:
            continue
        t, u = np.linalg.solve(np.column_stack([d, -e]), a)
        if t >= 0 and -1e-12 <= u <= 1 + 1e-12:
            best = max(best, float(t))
    return best


def tripping_mva(poly: list[Point], v_ll_kv: float, phi_deg: float, factor: float = 1.0) -> float:
    """Smallest |S| along a loading path at power angle phi for which the apparent impedance enters the zone."""
    zb = boundary_distance(poly, np.radians(phi_deg))
    return v_ll_kv**2 / (factor * zb) if zb > 0 else np.inf


def quadrilateral_zone(x_reach_ohm: float, r_reach_ohm: float) -> list[Point]:
    """Exercise 5: a forward quadrilateral zone, reactance reach ``x_reach`` and resistive blinder ``r_reach``."""
    return [(0.0, 0.0), (r_reach_ohm, 0.0), (r_reach_ohm, x_reach_ohm), (0.0, x_reach_ohm)]


def infeed_apparent_impedance(z_line: complex, position: float, r_fault: float, remote_share: float) -> complex:
    """Relay impedance for a fault at ``position`` of the line with fault resistance and a remote infeed.

    With I_remote = remote_share / (1 - remote_share) times the relay current, the fault resistance is seen
    magnified: Z = position Z_line + R_f (I_relay + I_remote) / I_relay.
    """
    magnification = 1.0 / (1.0 - remote_share)
    return position * z_line + r_fault * magnification


def main() -> None:
    """Lab Part B and Exercise 5 numbers."""
    poly = toop_polygon(30.0, 2.5, 10.0, 15.0)
    vertices = [(round(float(r), 3), round(float(x), 3)) for r, x in poly]
    print('ToOp polygon (30 deg):', vertices, 'valid:', is_valid(30.0, 2.5, 10.0, 15.0), 'simple:', is_simple(poly))
    bad = toop_polygon(80.0, 2.5, 10.0, 15.0)
    print(
        f'80 deg: r_v tan(a) = {10 * np.tan(np.radians(80)):.2f} > x_v = 15 -> valid:',
        is_valid(80.0, 2.5, 10.0, 15.0),
        end=' ',
    )
    print('simple:', is_simple(bad))
    v = 380.0
    for s_mva in (1000.0, 5000.0, 10000.0, 14000.0):
        z = apparent_impedance(v, s_mva * 0.95, s_mva * np.sqrt(1 - 0.95**2))
        print(f'  S = {s_mva:6.0f} MVA at cos phi 0.95: Z = ({z[0]:7.2f}, {z[1]:6.2f}) ohm,', end=' ')
        print(f'in zone: {trips(poly, z)}, with factor 1.41: {trips(poly, z, 1.41)}')
    phi = np.degrees(np.arccos(0.95))
    print(f'tripping MVA at cos phi 0.95 (phi = {phi:.1f} deg): {tripping_mva(poly, v, phi):.0f} MVA;', end=' ')
    print(f'with the 1.41 warning factor: {tripping_mva(poly, v, phi, 1.41):.0f} MVA')
    print(f'tripping MVA at cos phi 1.0: {tripping_mva(poly, v, 0.0):.0f} MVA;', end=' ')
    print(f'at phi = 60 deg: {tripping_mva(poly, v, 60.0):.0f} MVA')
    z_rev = apparent_impedance(v, -60000.0 * 0.95, -60000.0 * np.sqrt(1 - 0.95**2))
    print(f'reverse flow 60000 MVA: Z = ({z_rev[0]:.2f}, {z_rev[1]:.2f});', end=' ')
    print(f'raw point inside: {contains(poly, z_rev)}, ToOp (|R|, |X|): {trips(poly, z_rev)}')

    z_line = 100 * complex(0.03, 0.3)
    zone1, zone2 = quadrilateral_zone(0.8 * z_line.imag, 10.0), quadrilateral_zone(1.2 * z_line.imag, 10.0)
    print(f'Exercise 5: Z_line = {z_line} ohm; zone 1 reach X = {0.8 * z_line.imag} ohm,', end=' ')
    print(f'zone 2 reach X = {1.2 * z_line.imag:.0f} ohm, blinders at 10 ohm')
    for share in (0.0, 0.5):
        z = infeed_apparent_impedance(z_line, 0.9, 5.0, share)
        print(
            f'  fault at 90 %, R_f = 5 ohm, remote share {share:.0%}: Z = ({z.real:.2f}, {z.imag:.2f}) ohm,', end=' '
        )
        print(f'zone 1: {contains(zone1, (z.real, z.imag))}, zone 2: {contains(zone2, (z.real, z.imag))}')


if __name__ == '__main__':
    main()
