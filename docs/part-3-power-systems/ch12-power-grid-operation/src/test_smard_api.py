import urllib.error
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import smard_api
from smard_api import (
    CONSUMPTION_FILTERS,
    GENERATION_FILTERS,
    LIVE_WEEK,
    SmardUnavailableError,
    available_weeks,
    fetch_json,
    fetch_series,
    fetch_table,
    live_or_saved,
    live_paths,
    load_week,
    save_week,
    to_berlin,
    week_timestamp,
)
from smard_week import CONSUMPTION_COLUMNS, GENERATION_COLUMNS, HOURS, LOAD, critical_hours, residual_load

if TYPE_CHECKING:
    from pathlib import Path

LIVE_WEEK_MS = 1768172400000  # Monday 12 January 2026 00:00 Berlin time, from SMARD's index_hour.json


def test_live_week_timestamp_is_berlin_midnight():
    assert week_timestamp(LIVE_WEEK) == LIVE_WEEK_MS
    assert to_berlin([LIVE_WEEK_MS, LIVE_WEEK_MS + 3_600_000]).tolist() == [
        LIVE_WEEK,
        LIVE_WEEK + pd.Timedelta(hours=1),
    ]
    with pytest.raises(ValueError, match='not a Monday'):
        week_timestamp(pd.Timestamp('2026-01-13'))


def test_filters_cover_the_export_columns():
    assert list(GENERATION_FILTERS) == GENERATION_COLUMNS
    assert list(CONSUMPTION_FILTERS) == CONSUMPTION_COLUMNS


def test_saved_week_has_every_hour_and_column():
    gen, cons = load_week()
    assert list(gen.columns) == GENERATION_COLUMNS and list(cons.columns) == CONSUMPTION_COLUMNS
    assert len(gen) == len(cons) == HOURS
    assert gen.index[0] == LIVE_WEEK and gen.index[-1] == pd.Timestamp('2026-01-18 23:00')
    assert (gen.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all() and gen.index.equals(cons.index)
    assert gen['Nuclear'].isna().all()
    assert not gen.drop(columns='Nuclear').isna().any().any() and not cons.isna().any().any()


def test_saved_week_residual_load_recomputes():
    gen, cons = load_week()
    assert np.abs(residual_load(gen, cons) - cons['Residual load']).max() <= 0.02
    assert cons[LOAD].between(45_000, 76_000).all()


def test_saved_week_critical_hours():
    gen, cons = load_week()
    crit = critical_hours(gen, cons)
    assert len(crit) == 19
    assert crit.index.min() == pd.Timestamp('2026-01-12 05:00') and crit.index.max() == pd.Timestamp(
        '2026-01-16 06:00'
    )
    assert crit['residual'].idxmin() == pd.Timestamp('2026-01-16 02:00') and round(crit['residual'].min()) == 13452


def test_save_and_load_round_trip(tmp_path: Path):
    gen, cons = load_week()
    paths = save_week(gen, cons, tmp_path)
    assert paths == live_paths(LIVE_WEEK, tmp_path) and paths[0].name == 'smard_live_week_2026-01-12_generation.csv'
    back_gen, back_cons = load_week(LIVE_WEEK, tmp_path)
    assert back_gen.equals(gen) and back_cons.equals(cons)


def test_fetch_json_maps_network_errors(monkeypatch: pytest.MonkeyPatch):
    def offline(url, timeout):
        raise urllib.error.URLError('no network')

    monkeypatch.setattr(smard_api.urllib.request, 'urlopen', offline)
    with pytest.raises(SmardUnavailableError, match='cannot reach SMARD'):
        fetch_json('https://www.smard.de/x')

    def no_tls(url, timeout):  # what urlopen raises in Pyodide (JupyterLite)
        raise RuntimeError('TLS not supported in this environment')

    monkeypatch.setattr(smard_api.urllib.request, 'urlopen', no_tls)
    with pytest.raises(SmardUnavailableError, match='cannot reach SMARD'):
        fetch_json('https://www.smard.de/x')

    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, 'Not Found', None, None)

    monkeypatch.setattr(smard_api.urllib.request, 'urlopen', not_found)
    assert fetch_json('https://www.smard.de/x') is None
    assert fetch_series(1224, LIVE_WEEK) is None


def test_fetch_table_fills_a_missing_filter_with_nan(monkeypatch: pytest.MonkeyPatch):
    gen, _ = load_week()

    def from_saved(filter_id, monday, region):
        return None if filter_id == 1224 else gen[next(n for n, f in GENERATION_FILTERS.items() if f == filter_id)]

    monkeypatch.setattr(smard_api, 'fetch_series', from_saved)
    table = fetch_table(GENERATION_FILTERS, LIVE_WEEK)
    assert list(table.columns) == GENERATION_COLUMNS and table['Nuclear'].isna().all()
    assert table.index.name == 'start' and table.drop(columns='Nuclear').equals(gen.drop(columns='Nuclear'))


def test_live_or_saved_falls_back_offline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    def offline(url, timeout=None):
        raise SmardUnavailableError('cannot reach SMARD (test)')

    monkeypatch.setattr(smard_api, 'fetch_json', offline)
    gen, cons, source = live_or_saved()
    assert source == 'saved' and len(gen) == len(cons) == HOURS
    assert 'using the saved week' in capsys.readouterr().out


@pytest.mark.network
def test_live_index_and_series_are_plausible():
    weeks = available_weeks()
    assert (weeks.weekday == 0).all() and (weeks.hour == 0).all()
    assert weeks[0] == pd.Timestamp('2014-12-29') and LIVE_WEEK in weeks
    load = fetch_series(410, LIVE_WEEK)
    assert len(load) == HOURS and load.index[0] == LIVE_WEEK and load.between(30_000, 90_000).all()
    saved = load_week()[1][LOAD]
    assert abs(load.mean() - saved.mean()) < 0.01 * saved.mean()  # SMARD revises data, so not an exact match
