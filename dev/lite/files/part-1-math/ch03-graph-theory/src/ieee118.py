"""The IEEE 118-bus branch list from pandapower, run through the same functions as the 14-bus grid (lab step 6)."""

import grid_graph
import pandapower.networks as pn
import pandas as pd


def edges_from_pandapower(net) -> pd.DataFrame:
    """Branch list ``from_bus, to_bus`` of a pandapower net (lines and transformers), bus numbers from 1."""
    lines = net.line[['from_bus', 'to_bus']]
    trafos = net.trafo[['hv_bus', 'lv_bus']].rename(columns={'hv_bus': 'from_bus', 'lv_bus': 'to_bus'})
    return pd.concat([lines, trafos], ignore_index=True).astype(int) + 1


def main() -> None:
    """Print the bridges, articulation points and Laplacian figures of the 118-bus grid quoted in the solution."""
    edges = edges_from_pandapower(pn.case118())
    graph = grid_graph.build_graph(edges)
    print(f'{len(edges)} branches, {graph.number_of_nodes()} buses')
    bridges, points = grid_graph.bridges_and_articulation_points(graph)
    print(f'{len(bridges)} bridges:', bridges)
    print(f'{len(points)} articulation points:', points)
    print('zero eigenvalues:', grid_graph.count_zero_eigenvalues(graph))
    reff = grid_graph.effective_resistances(graph)
    print('branches with R_eff = 1:', len(reff[reff['r_eff'] > 1 - 1e-9]))
    print(f'algebraic connectivity: {grid_graph.algebraic_connectivity(graph):.4f}')


if __name__ == '__main__':
    main()
