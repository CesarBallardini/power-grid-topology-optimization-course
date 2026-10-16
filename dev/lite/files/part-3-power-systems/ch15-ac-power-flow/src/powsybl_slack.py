"""Single versus distributed slack on IEEE 14 with pypowsybl (Chapter 15, Lab step 5 and Exercise 6).

pypowsybl reports a generator's active power with the receptor sign convention (negative when generating), so
the module negates it to print MW produced. Branch flows ``p1`` are the active power entering at side 1.
"""

import pandas as pd
import pypowsybl as pyp
import pypowsybl.loadflow as lf

BALANCE = {
    'single slack': lf.Parameters(distributed_slack=False),
    'proportional to P': lf.Parameters(
        distributed_slack=True, balance_type=lf.BalanceType.PROPORTIONAL_TO_GENERATION_P
    ),
    'proportional to Pmax': lf.Parameters(
        distributed_slack=True, balance_type=lf.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX
    ),
}


def solve_ieee14(parameters: lf.Parameters, load_factor: float = 1.0) -> tuple[pd.Series, pd.Series, float]:
    """Generator outputs (MW), branch flows ``p1`` (MW) and the remaining slack mismatch of an AC load flow.

    The stock case is balanced (the generator targets already include the losses), so ``load_factor`` scales every
    load to create a mismatch for the slack to absorb.
    """
    net = pyp.network.create_ieee14()
    loads = net.get_loads()
    net.update_loads(id=loads.index, p0=loads['p0'] * load_factor, q0=loads['q0'] * load_factor)
    result = lf.run_ac(net, parameters)[0]
    if result.status != lf.ComponentStatus.CONVERGED:
        raise RuntimeError(f'load flow did not converge: {result.status}')
    gens = -net.get_generators()['p']
    flows = pd.concat([net.get_lines()['p1'], net.get_2_windings_transformers()['p1']])
    return gens, flows, result.slack_bus_results[0].active_power_mismatch


def compare(load_factor: float = 1.0, cases: dict[str, lf.Parameters] = BALANCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generator outputs and the five largest base-case branch flows, one column per slack setting."""
    gens, flows = {}, {}
    for name, parameters in cases.items():
        gens[name], flows[name], _ = solve_ieee14(parameters, load_factor)
    gens, flows = pd.DataFrame(gens), pd.DataFrame(flows)
    top = flows.abs().max(axis=1).nlargest(5).index
    return gens, flows.loc[top]


def main() -> None:
    """Print the tables of Lab step 5 and Exercise 6."""
    pd.set_option('display.width', 120)
    targets = pyp.network.create_ieee14().get_generators()['target_p']
    for load_factor in (1.0, 1.1):
        print(f'== loads scaled by {load_factor} ==')
        gens, flows = compare(load_factor)
        gens.insert(0, 'target_p', targets)
        print('generator active power (MW)')
        print(gens.round(3))
        print('total generation (MW):', gens.drop(columns='target_p').sum().round(3).to_dict())
        print('five largest base-case flows p1 (MW)')
        print(flows.round(3))
        print(
            'largest change of any branch flow against the single slack (MW):',
            flows.sub(flows['single slack'], axis=0).abs().max().round(3).to_dict(),
        )
        print(
            'remaining slack mismatch (MW):',
            {k: round(solve_ieee14(v, load_factor)[2], 3) for k, v in BALANCE.items()},
        )


if __name__ == '__main__':
    main()
