"""IEEE 14 as a DC model for the labs of Chapter 18: nodes, branches, susceptances and injections from pandapower.

``load_case14()`` reads pandapower's ``case14`` after a DC power flow and returns a ``DcGrid``. The grid can be
split at a node, lose branches or change a susceptance, each time returning a new ``DcGrid`` whose PTDF is
computed from scratch: the brute-force references against which the low-rank updates of the chapter are checked.
Flows are in MW, susceptances in per unit on the system base, bus 0 is the slack.
"""

from __future__ import annotations  # ToOp's interpreter is older than 3.14: methods return the class they live in

from dataclasses import dataclass, replace

import numpy as np
import pandapower as pp
import pandapower.networks as pn


def ptdf(frm: np.ndarray, to: np.ndarray, b: np.ndarray, n_bus: int, slack: int = 0) -> np.ndarray:
    """PTDF (branches x buses) of a DC grid, with zero columns for the slack and for isolated nodes."""
    n_br = len(frm)
    a = np.zeros((n_br, n_bus))
    a[np.arange(n_br), frm], a[np.arange(n_br), to] = 1.0, -1.0
    bf = b[:, None] * a
    bbus = a.T @ bf
    degree = np.bincount(np.concatenate([frm, to]), minlength=n_bus)
    keep = (np.arange(n_bus) != slack) & (degree > 0)  # nodes left without branches (busbar outages) are dropped
    h = np.zeros((n_br, n_bus))
    h[:, keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], bf[:, keep].T).T
    return h


@dataclass(frozen=True)
class DcGrid:
    """A DC grid: from/to nodes, susceptances ``b``, nodal injections ``p`` (MW) and branch names."""

    frm: np.ndarray
    to: np.ndarray
    b: np.ndarray
    p: np.ndarray
    names: list[str]
    slack: int = 0

    @property
    def n_bus(self) -> int:
        """Number of nodes, including nodes added by splits."""
        return len(self.p)

    @property
    def n_branch(self) -> int:
        """Number of branches."""
        return len(self.frm)

    def ptdf(self) -> np.ndarray:
        """PTDF computed from scratch (the reference the updates are compared with)."""
        return ptdf(self.frm, self.to, self.b, self.n_bus, self.slack)

    def flows(self) -> np.ndarray:
        """Branch flows ``H P`` in MW."""
        return self.ptdf() @ self.p

    def branches_at(self, node: int) -> list[int]:
        """Indices of the branches with one end at ``node``."""
        return [k for k in range(self.n_branch) if self.frm[k] == node or self.to[k] == node]

    def split(self, node: int, bus_a: list[int]) -> DcGrid:
        """Physically split ``node``: ``bus_a`` branches stay, the others move to a new node; injections stay."""
        nb = self.n_bus
        frm, to = self.frm.copy(), self.to.copy()
        for k in self.branches_at(node):
            if k not in bus_a:
                if frm[k] == node:
                    frm[k] = nb
                else:
                    to[k] = nb
        return replace(self, frm=frm, to=to, p=np.append(self.p, 0.0))

    def without(self, branches: list[int]) -> DcGrid:
        """The grid with the given branches removed (their PTDF rows are gone, use ``names`` to map back)."""
        keep = [k for k in range(self.n_branch) if k not in branches]
        return replace(self, frm=self.frm[keep], to=self.to[keep], b=self.b[keep], names=[self.names[k] for k in keep])

    def scaled(self, branch: int, factor: float) -> DcGrid:
        """The grid with the susceptance of one branch multiplied by ``factor``."""
        b = self.b.copy()
        b[branch] *= factor
        return replace(self, b=b)

    def moved(self, node_from: int, node_to: int, mw: float) -> DcGrid:
        """The grid with ``mw`` of injection moved from one node to another."""
        p = self.p.copy()
        p[node_from] -= mw
        p[node_to] += mw
        return replace(self, p=p)


def ratings(grid: DcGrid, margin: float = 1.3, floor: float = 40.0) -> np.ndarray:
    """MW limits for a case that ships without any: ``margin`` times the base-case flow, at least ``floor`` MW."""
    return np.maximum(margin * np.abs(grid.flows()), floor)


def from_pandapower(net: pp.pandapowerNet) -> DcGrid:
    """The DC model of a pandapower network with contiguous bus indices and the ``ext_grid`` at bus 0."""
    pp.rundcpp(net)
    branch, bus, gen = net._ppc['branch'], net._ppc['bus'], net._ppc['gen']
    frm, to = branch[:, 0].real.astype(int), branch[:, 1].real.astype(int)
    tap = np.where(branch[:, 8].real == 0, 1.0, branch[:, 8].real)
    b = 1.0 / (branch[:, 3].real * tap)
    p = -bus[:, 2].real.copy()  # loads
    np.add.at(p, gen[:, 0].real.astype(int), gen[:, 1].real)  # generators, slack included
    p[0] -= p.sum()  # the slack balances the losses pandapower assigns to it
    names = [f'L{i + 1}-{j + 1}' for i, j in zip(frm, to, strict=True)]
    for k in range(len(net.line), len(frm)):
        names[k] = 'T' + names[k][1:]
    return DcGrid(frm=frm, to=to, b=b, p=p, names=names)


def load_case14() -> DcGrid:
    """IEEE 14 from pandapower: 20 branches (15 lines, 5 transformers), per-unit reactances from the ppc tables."""
    return from_pandapower(pn.case14())


def main() -> None:
    """Print the base-case flows and check them against pandapower."""
    grid = load_case14()
    net = pn.case14()
    pp.rundcpp(net)
    ref = np.concatenate([net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()])
    f = grid.flows()
    print('largest difference to pandapower rundcpp:', np.abs(f - ref).max(), 'MW')
    for name, flow in zip(grid.names, f, strict=True):
        print(f'{name:7s} {flow:8.2f} MW')


if __name__ == '__main__':
    main()
