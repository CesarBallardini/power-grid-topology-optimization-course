"""Fetch one week of hourly SMARD generation and consumption through the public JSON API (Chapter 12).

The download center of SMARD (https://www.smard.de) is a form, but the data behind its charts is a JSON API that
needs no login: ``.../chart_data/{filter}/{region}/index_hour.json`` lists the weeks available for a filter (ms
since the epoch, each a Monday 00:00 Berlin time) and ``.../chart_data/{filter}/{region}/{filter}_{region}_hour_
{timestamp}.json`` returns ``{"meta_data": ..., "series": [[timestamp, value], ...]}``, one value per hour of that
week in MWh. The filter ids are the ones documented by the Bundesnetzagentur (github.com/bundesAPI/smard-api) and
were checked against the data: the residual-load filter equals grid load minus wind and solar to 0.01 MWh, and the
values are those of the CSV export of the same week. A filter without data in the week (nuclear, since April 2023)
answers 404 and becomes a column of NaN, where the export prints ``-``.

``fetch_week`` returns the two tables with the columns of the CSV export, so every function of ``smard_week``
processes them unchanged; ``save_week`` writes them in the export layout next to this file, and ``load_week``
reads them back, so the lab runs offline on the saved week of 12–18 January 2026.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from smard_week import LOAD, read_smard, report, to_smard_csv

HERE = Path(__file__).resolve().parent
BASE_URL = 'https://www.smard.de/app/chart_data'
REGION = 'DE'
TIMEOUT_S = 30
TIMEZONE = 'Europe/Berlin'
GENERATION_FILTERS = {
    'Biomass': 4066,
    'Hydropower': 1226,
    'Wind offshore': 1225,
    'Wind onshore': 4067,
    'Photovoltaics': 4068,
    'Other renewable': 1228,
    'Nuclear': 1224,
    'Lignite': 1223,
    'Hard coal': 4069,
    'Fossil gas': 4071,
    'Hydro pumped storage': 4070,
    'Other conventional': 1227,
}
CONSUMPTION_FILTERS = {LOAD: 410, 'Hydro pumped storage': 4387, 'Residual load': 4359}
LIVE_WEEK = pd.Timestamp('2026-01-12')  # the Monday of the week saved next to this file


class SmardUnavailableError(ConnectionError):
    """The SMARD API could not be reached: no network, a timeout or a server error."""


def fetch_json(url: str, timeout: float = TIMEOUT_S) -> dict | None:
    """The JSON document at ``url``, or ``None`` when SMARD answers 404 (no data for that filter and week)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 -- https URL under BASE_URL
            return json.load(response)
    except urllib.error.HTTPError as error:
        error.close()  # the error carries the response body, a file that would otherwise stay open
        if error.code == 404:
            return None
        raise SmardUnavailableError(f'SMARD answered HTTP {error.code} for {url}') from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise SmardUnavailableError(
            f'cannot reach SMARD ({error}); check the network connection, then retry'
        ) from error


def to_berlin(timestamps_ms) -> pd.DatetimeIndex:
    """Epoch milliseconds as naive Berlin local timestamps, the clock the SMARD exports use."""
    return pd.to_datetime(list(timestamps_ms), unit='ms', utc=True).tz_convert(TIMEZONE).tz_localize(None)


def week_timestamp(monday: pd.Timestamp) -> int:
    """Epoch milliseconds of ``monday`` 00:00 Berlin time, the key under which SMARD files that week."""
    if monday.weekday() != 0 or monday != monday.normalize():
        raise ValueError(f'{monday} is not a Monday at 00:00')
    return int(monday.tz_localize(TIMEZONE).timestamp() * 1000)


def available_weeks(filter_id: int = 410, region: str = REGION) -> pd.DatetimeIndex:
    """The Mondays of every week SMARD has for a filter, oldest first."""
    index = fetch_json(f'{BASE_URL}/{filter_id}/{region}/index_hour.json')
    if index is None:
        raise SmardUnavailableError(f'SMARD has no hourly index for filter {filter_id} in {region}')
    return to_berlin(index['timestamps'])


def fetch_series(filter_id: int, monday: pd.Timestamp, region: str = REGION) -> pd.Series | None:
    """The hourly values (MWh) of one filter for the week of ``monday``, or ``None`` when SMARD has none."""
    stamp = week_timestamp(monday)
    data = fetch_json(f'{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_hour_{stamp}.json')
    if data is None:
        return None
    timestamps, values = zip(*data['series'], strict=True)
    return pd.Series(values, index=to_berlin(timestamps), dtype=float)


def fetch_table(filters: dict[str, int], monday: pd.Timestamp, region: str = REGION) -> pd.DataFrame:
    """One export table: a column per filter, in the given order, NaN for filters without data."""
    series = {name: fetch_series(filter_id, monday, region) for name, filter_id in filters.items()}
    table = pd.DataFrame({name: s for name, s in series.items() if s is not None}).reindex(columns=list(filters))
    table.index.name = 'start'
    return table


def fetch_week(monday: pd.Timestamp = LIVE_WEEK, region: str = REGION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation by source and consumption for the week of ``monday``, in the layout of the SMARD exports."""
    return fetch_table(GENERATION_FILTERS, monday, region), fetch_table(CONSUMPTION_FILTERS, monday, region)


def live_paths(monday: pd.Timestamp = LIVE_WEEK, folder: Path = HERE) -> tuple[Path, Path]:
    """Where the generation and consumption tables of the week of ``monday`` are saved."""
    day = monday.strftime('%Y-%m-%d')
    return folder / f'smard_live_week_{day}_generation.csv', folder / f'smard_live_week_{day}_consumption.csv'


def save_week(gen: pd.DataFrame, cons: pd.DataFrame, folder: Path = HERE) -> tuple[Path, Path]:
    """Write the two tables in the export layout (``smard_week.to_smard_csv``) and return their paths."""
    paths = live_paths(gen.index[0], folder)
    for path, table in zip(paths, (gen, cons), strict=True):
        path.write_text(to_smard_csv(table), encoding='utf-8', newline='\n')
    return paths


def load_week(monday: pd.Timestamp = LIVE_WEEK, folder: Path = HERE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The saved tables of the week of ``monday``, read with ``smard_week.read_smard``."""
    gen_path, cons_path = live_paths(monday, folder)
    return read_smard(gen_path), read_smard(cons_path)


def live_or_saved(monday: pd.Timestamp = LIVE_WEEK) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The week from the API when SMARD answers, otherwise the saved copy; the third item says which."""
    try:
        gen, cons = fetch_week(monday)
    except SmardUnavailableError as error:
        print(f'{error}; using the saved week')
        gen, cons = load_week(monday)
        return gen, cons, 'saved'
    return gen, cons, 'api'


def main() -> None:
    """Fetch the live week, save it next to this file and print the numbers quoted in the solutions."""
    gen, cons = fetch_week(LIVE_WEEK)
    weeks = available_weeks()
    print(f'SMARD has {len(weeks)} weeks of hourly data, {weeks[0].date()} to {weeks[-1].date()}')
    print('no data (NaN):', [c for c in gen.columns if gen[c].isna().all()])
    gen_path, cons_path = save_week(gen, cons)
    print('saved', gen_path.name, 'and', cons_path.name)
    saved_gen, saved_cons = load_week(LIVE_WEEK)
    print('saved files reproduce the API week:', saved_gen.equals(gen.round(2)) and saved_cons.equals(cons.round(2)))
    report(gen, cons)


if __name__ == '__main__':
    main()
