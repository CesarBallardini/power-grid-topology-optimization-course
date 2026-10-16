"""Voltage levels and switches of pypowsybl's four-substations node-breaker network (Chapter 13, Exercise 5).

Every voltage level of this example is modelled node-breaker, so each one has a list of switches with a kind
(``BREAKER`` or ``DISCONNECTOR``), an open flag and the two nodes it joins. In the bay of an asset, the breaker
interrupts current and the busbar disconnectors select the busbar; in a double-busbar level the coupler is a breaker
between the two busbar sections, with a disconnector on each side.
"""

import pandas as pd
import pypowsybl as pp

VOLTAGE_LEVEL = 'S1VL2'  # the only double-busbar level of the example: two busbar sections and a coupler


def network() -> pp.network.Network:
    """The four-substations node-breaker example shipped with pypowsybl."""
    return pp.network.create_four_substations_node_breaker_network()


def voltage_levels(net: pp.network.Network) -> pd.DataFrame:
    """Id, substation, nominal voltage and topology kind of every voltage level."""
    return net.get_voltage_levels(attributes=['substation_id', 'nominal_v', 'topology_kind'])


def switch_counts(net: pp.network.Network) -> pd.DataFrame:
    """Number of switches per voltage level and kind."""
    return net.get_switches().groupby(['voltage_level_id', 'kind']).size().unstack(fill_value=0)


def classify_switches(net: pp.network.Network, voltage_level: str = VOLTAGE_LEVEL) -> pd.DataFrame:
    """The switches of one voltage level with a ``role`` column.

    Roles: ``coupler`` (the breaker between the two busbar sections), ``coupler disconnector`` (its busbar
    selectors), ``bay breaker`` (the breaker of an asset bay) and ``busbar disconnector`` (the selector of an asset
    bay onto one busbar section). The role follows from the kind and from the node the switch touches: a switch
    touching a busbar-section node is a selector.
    """
    topo = net.get_node_breaker_topology(voltage_level)
    busbar_nodes = set(topo.nodes.index[topo.nodes['connectable_type'] == 'BUSBAR_SECTION'])
    switches = topo.switches.copy()
    on_busbar = switches['node1'].isin(busbar_nodes) | switches['node2'].isin(busbar_nodes)
    is_coupler = switches.index.str.contains('COUPLER')
    role = pd.Series('bay breaker', index=switches.index)
    role[(switches['kind'] == 'DISCONNECTOR') & on_busbar & ~is_coupler] = 'busbar disconnector'
    role[(switches['kind'] == 'DISCONNECTOR') & is_coupler] = 'coupler disconnector'
    role[(switches['kind'] == 'BREAKER') & is_coupler] = 'coupler'
    return switches.assign(role=role)


def main() -> None:
    """Print the voltage levels, the switch counts and the classification of the switches of S1VL2."""
    pd.set_option('display.width', 119)
    net = network()
    print(voltage_levels(net))
    print()
    print(switch_counts(net))
    print()
    classified = classify_switches(net)
    print(classified[['kind', 'open', 'retained', 'node1', 'node2', 'role']])
    print()
    print(classified.groupby(['kind', 'role']).size())


if __name__ == '__main__':
    main()
