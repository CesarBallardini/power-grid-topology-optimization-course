"""Parse a week of SMARD generation and consumption exports and derive the residual load (Chapter 12).

The SMARD download center writes one CSV per table (generation by source; consumption), with a semicolon as
delimiter, a UTF-8 byte-order mark, two timestamp columns for the start and the end of each interval, ``-`` where
a value is missing (nuclear, since April 2023) and every value column named ``<source> [MWh] Calculated
resolutions``. Dates and numbers follow the language of the portal: the English export writes
``Jan 12, 2026 12:00 AM`` and ``12,345.67``; the German export (``Datum von``, German column names, suffix
``Berechnete Auflösungen``) writes ``12.01.2026 00:00`` and ``12.345,67``. ``read_smard`` tells the two apart by the
first header cell and returns the English short names either way; both layouts were checked against the real
exports of the week of 12 January 2026. ``to_smard_csv`` writes the English layout.

The committed samples ``smard_generation_sample.csv`` and ``smard_consumption_sample.csv`` are synthetic:
``write_samples`` builds a plausible winter week (a weekday daily cycle, a lower weekend, a wind front passing
mid-week, short solar days) in exactly this layout, so that the parsing code can be exercised and tested without a
download. ``smard_api`` fetches a real week through the SMARD JSON API and saves it in the same layout.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GENERATION_CSV = HERE / 'smard_generation_sample.csv'
CONSUMPTION_CSV = HERE / 'smard_consumption_sample.csv'
WEEK_START = pd.Timestamp('2026-01-12 00:00')  # a Monday
HOURS = 7 * 24
SUFFIX = ' [MWh] Calculated resolutions'
GENERATION_COLUMNS = [
    'Biomass',
    'Hydropower',
    'Wind offshore',
    'Wind onshore',
    'Photovoltaics',
    'Other renewable',
    'Nuclear',
    'Lignite',
    'Hard coal',
    'Fossil gas',
    'Hydro pumped storage',
    'Other conventional',
]
LOAD = 'grid load'  # the consumption table's name for the total load, in the order of the export
CONSUMPTION_COLUMNS = [LOAD, 'Hydro pumped storage', 'Residual load']
RENEWABLE_VARIABLE = ['Wind offshore', 'Wind onshore', 'Photovoltaics']
GERMAN_NAMES = {
    'Biomasse': 'Biomass',
    'Wasserkraft': 'Hydropower',
    'Wind Offshore': 'Wind offshore',
    'Wind Onshore': 'Wind onshore',
    'Photovoltaik': 'Photovoltaics',
    'Sonstige Erneuerbare': 'Other renewable',
    'Kernenergie': 'Nuclear',
    'Braunkohle': 'Lignite',
    'Steinkohle': 'Hard coal',
    'Erdgas': 'Fossil gas',
    'Pumpspeicher': 'Hydro pumped storage',
    'Sonstige Konventionelle': 'Other conventional',
    'Netzlast': LOAD,
    'Residuallast': 'Residual load',
}
# The two languages of the portal, keyed by the first header cell: the end column, the date format, the thousands
# and decimal marks, the suffix of every value column and the column names to translate.
LAYOUTS = {
    'Start date': ('End date', '%b %d, %Y %I:%M %p', ',', '.', SUFFIX, {}),
    'Datum von': ('Datum bis', '%d.%m.%Y %H:%M', '.', ',', ' [MWh] Berechnete Auflösungen', GERMAN_NAMES),
}


def synthetic_week() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation by source and consumption, hourly MWh, for one synthetic winter week.

    The shapes are deterministic formulas of the hour of the week: consumption has a daytime hump and an evening
    peak, scaled down at the weekend; onshore wind is a bell-shaped front centred on Wednesday night on top of a
    slow swell; offshore wind follows onshore at a smaller scale; photovoltaics is a short daytime arc whose
    height changes from day to day; the thermal fleet covers what is left above a must-run floor.
    """
    t = np.arange(HOURS)
    hour, day = t % 24, t // 24
    weekend = np.where(day >= 5, 0.86, 1.0)
    load = weekend * (
        47000 + 13000 * np.exp(-(((hour - 12) / 5.0) ** 2)) + 4000 * np.exp(-(((hour - 18.5) / 2.0) ** 2))
    )
    wind_on = np.maximum(500.0, 8000 + 30000 * np.exp(-(((t - 80) / 30.0) ** 2)) + 6000 * np.sin(2 * np.pi * t / 40))
    wind_off = np.minimum(9000.0, 1500 + 0.22 * wind_on)
    solar_day = np.array([0.6, 1.0, 0.3, 0.5, 0.9, 0.7, 0.4])[day]
    solar = 11000 * solar_day * np.maximum(0.0, np.sin(np.pi * (hour - 8) / 9.0)) * ((hour >= 8) & (hour <= 17))
    biomass, hydro, other_ren = np.full(HOURS, 4800.0), np.full(HOURS, 1600.0), np.full(HOURS, 150.0)
    residual = load - wind_on - wind_off - solar
    pumped = np.maximum(0.0, residual - 45000) * 0.15
    thermal = np.maximum(residual - pumped - biomass - hydro - other_ren, 7000.0)
    gen = pd.DataFrame(
        {
            'Biomass': biomass,
            'Hydropower': hydro,
            'Wind offshore': wind_off,
            'Wind onshore': wind_on,
            'Photovoltaics': solar,
            'Other renewable': other_ren,
            'Nuclear': np.full(HOURS, np.nan),  # no nuclear generation in Germany since April 2023: SMARD prints '-'
            'Lignite': 0.45 * thermal,
            'Hard coal': 0.20 * thermal,
            'Fossil gas': 0.30 * thermal,
            'Hydro pumped storage': pumped,
            'Other conventional': 0.05 * thermal,
        },
        index=pd.date_range(WEEK_START, periods=HOURS, freq='h'),
    )
    cons = pd.DataFrame({LOAD: load, 'Hydro pumped storage': 1.1 * pumped, 'Residual load': residual}, index=gen.index)
    return gen, cons


def english_stamp(t: pd.Timestamp) -> str:
    """A timestamp as the English export writes it, ``Jan 5, 2026 1:00 AM``: day and hour without a leading zero."""
    return f'{t:%b} {t.day}, {t.year} {t.hour % 12 or 12}:{t.minute:02d} {"AM" if t.hour < 12 else "PM"}'


def to_smard_csv(table: pd.DataFrame) -> str:
    """Serialize a table in the layout of the English SMARD export: start and end columns, ``-`` for NaN."""
    lines = ['Start date;End date;' + ';'.join(f'{c}{SUFFIX}' for c in table.columns)]
    for start, row in zip(table.index, table.itertuples(index=False), strict=True):
        cells = ['-' if np.isnan(v) else f'{v:,.2f}' for v in row]
        lines.append(';'.join([english_stamp(start), english_stamp(start + pd.Timedelta(hours=1)), *cells]))
    return '\n'.join(lines) + '\n'


def write_samples(folder: Path = HERE) -> tuple[Path, Path]:
    """Write the two sample CSV files into ``folder`` and return their paths."""
    gen, cons = synthetic_week()
    paths = (folder / GENERATION_CSV.name, folder / CONSUMPTION_CSV.name)
    for path, table in zip(paths, (gen, cons), strict=True):
        path.write_text(to_smard_csv(table), encoding='utf-8', newline='\n')
    return paths


def read_smard(path: Path) -> pd.DataFrame:
    """Read a SMARD CSV export (English or German) into a frame indexed by the start of each interval.

    The first header cell selects the layout (``Start date`` or ``Datum von``); the byte-order mark of a real
    download is skipped, ``-`` becomes NaN, the end column is dropped, the ``[MWh] ...`` suffix is removed and
    German column names are translated, so the result has the same short English names whatever the language.
    """
    with open(path, encoding='utf-8-sig') as file:
        first_cell = file.readline().split(';', 1)[0]
    if first_cell not in LAYOUTS:
        raise ValueError(f'{path}: not a SMARD export, the header starts with {first_cell!r}')
    end, date_format, thousands, decimal, suffix, names = LAYOUTS[first_cell]
    table = pd.read_csv(path, sep=';', thousands=thousands, decimal=decimal, na_values=['-'], encoding='utf-8-sig')
    table.index = pd.to_datetime(table.pop(first_cell), format=date_format)
    table.index.name = 'start'
    table = table.drop(columns=[end])
    table.columns = [names.get(c.removesuffix(suffix), c.removesuffix(suffix)) for c in table.columns]
    return table


def residual_load(gen: pd.DataFrame, cons: pd.DataFrame) -> pd.Series:
    """Consumption minus wind (onshore and offshore) and photovoltaics, hour by hour."""
    return (cons[LOAD] - gen[RENEWABLE_VARIABLE].sum(axis=1)).rename('residual')


def duration_curve(series: pd.Series) -> pd.Series:
    """The values of a series sorted from largest to smallest, indexed by the number of hours at or above each."""
    values = np.sort(series.to_numpy())[::-1]
    return pd.Series(values, index=np.arange(1, len(values) + 1), name=series.name)


def critical_hours(gen: pd.DataFrame, cons: pd.DataFrame, q: float = 0.75) -> pd.DataFrame:
    """Hours with wind above its ``q`` quantile and residual load below its ``1 - q`` quantile.

    These are the hours in which the north–south corridors of the German grid congest: much wind in the north,
    little demand left for the thermal plants of the south.
    """
    wind = gen[['Wind offshore', 'Wind onshore']].sum(axis=1).rename('wind')
    residual = residual_load(gen, cons)
    mask = (wind >= wind.quantile(q)) & (residual <= residual.quantile(1 - q))
    return pd.DataFrame({'wind': wind, 'consumption': cons[LOAD], 'residual': residual})[mask]


def daily_summary(gen: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    """Per day: share of wind and solar in consumption, maximum hourly ramp of the residual load, its minimum."""
    renewable = gen[RENEWABLE_VARIABLE].sum(axis=1)
    load = cons[LOAD]
    residual = residual_load(gen, cons)
    by_day = pd.Grouper(freq='D')
    return pd.DataFrame(
        {
            'weekday': load.groupby(by_day).first().index.day_name(),
            'share_pct': 100 * renewable.groupby(by_day).sum() / load.groupby(by_day).sum(),
            'max_ramp_mw_h': residual.diff().abs().groupby(by_day).max(),
            'min_residual_mw': residual.groupby(by_day).min(),
        }
    )


def report(gen: pd.DataFrame, cons: pd.DataFrame) -> None:
    """Print the numbers that the solutions of Lab steps 2–3 and Exercise 4 quote for a week of data."""
    pd.set_option('display.width', 119)
    print('generation columns:', list(gen.columns))
    print('consumption columns:', list(cons.columns), 'rows:', len(cons))
    residual = residual_load(gen, cons)
    print(f'residual load matches the SMARD column to {np.abs(residual - cons["Residual load"]).max():.3f} MWh')
    load = cons[LOAD]
    wind = gen[['Wind offshore', 'Wind onshore']].sum(axis=1)
    for name, series in (
        ('consumption', load),
        ('wind', wind),
        ('solar', gen['Photovoltaics']),
        ('residual', residual),
    ):
        print(f'{name} {series.min():,.0f} ({series.idxmin()}) to {series.max():,.0f} MWh/h ({series.idxmax()})')
    print(f'wind and solar supplied {100 * gen[RENEWABLE_VARIABLE].sum().sum() / load.sum():.1f} % of the week')
    curve = duration_curve(residual)
    print(
        f'duration curve: {curve.iloc[0]:,.0f} at 1 h, {curve.iloc[83]:,.0f} at 84 h, {curve.iloc[-1]:,.0f} at 168 h'
    )
    crit = critical_hours(gen, cons)
    print(f'wind at or above {wind.quantile(0.75):,.0f} and residual at or below {residual.quantile(0.25):,.0f}:')
    print(f'{len(crit)} critical hours, from {crit.index.min()} to {crit.index.max()}')
    print(crit.round(0))
    ramp = residual.diff()
    print(f'largest ramp {ramp[ramp.abs().idxmax()]:+,.0f} MW in the hour ending {ramp.abs().idxmax()}')
    print(daily_summary(gen, cons).round({'share_pct': 1, 'max_ramp_mw_h': 0, 'min_residual_mw': 0}))


def main() -> None:
    """Print the numbers of the committed synthetic samples."""
    report(read_smard(GENERATION_CSV), read_smard(CONSUMPTION_CSV))


if __name__ == '__main__':
    main()
