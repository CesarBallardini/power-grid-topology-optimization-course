from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from smard_week import (
    CONSUMPTION_COLUMNS,
    CONSUMPTION_CSV,
    GENERATION_COLUMNS,
    GENERATION_CSV,
    HOURS,
    LOAD,
    critical_hours,
    daily_summary,
    duration_curve,
    english_stamp,
    read_smard,
    residual_load,
    synthetic_week,
    to_smard_csv,
    write_samples,
)

if TYPE_CHECKING:
    from pathlib import Path

GERMAN_EXPORT = (
    'Datum von;Datum bis;Netzlast [MWh] Berechnete Auflösungen;Pumpspeicher [MWh] Berechnete Auflösungen;'
    'Residuallast [MWh] Berechnete Auflösungen\r\n'
    '12.01.2026 00:00;12.01.2026 01:00;53.862,40;2.861,22;30.224,71\r\n'
    '12.01.2026 01:00;12.01.2026 02:00;52.156,79;-;28.228,64\r\n'
)


def test_committed_samples_are_the_synthetic_week(tmp_path: Path):
    gen_path, cons_path = write_samples(tmp_path)
    assert gen_path.read_text(encoding='utf-8') == GENERATION_CSV.read_text(encoding='utf-8')
    assert cons_path.read_text(encoding='utf-8') == CONSUMPTION_CSV.read_text(encoding='utf-8')


def test_english_stamp_drops_leading_zeros_like_the_export():
    assert english_stamp(pd.Timestamp('2026-01-05 01:00')) == 'Jan 5, 2026 1:00 AM'
    assert english_stamp(pd.Timestamp('2026-01-12 00:00')) == 'Jan 12, 2026 12:00 AM'
    assert english_stamp(pd.Timestamp('2026-01-18 23:00')) == 'Jan 18, 2026 11:00 PM'


def test_english_export_layout_round_trips(tmp_path: Path):
    table = pd.DataFrame({'X': [1234.5, np.nan, 0.25]}, index=pd.date_range('2026-01-12', periods=3, freq='h'))
    text = to_smard_csv(table)
    assert text.splitlines()[0] == 'Start date;End date;X [MWh] Calculated resolutions'
    assert text.splitlines()[1] == 'Jan 12, 2026 12:00 AM;Jan 12, 2026 1:00 AM;1,234.50'
    assert text.splitlines()[2].endswith(';-')
    path = tmp_path / 'x.csv'
    path.write_text(text, encoding='utf-8')
    back = read_smard(path)
    assert list(back.columns) == ['X'] and np.isnan(back['X'].iloc[1]) and back['X'].iloc[2] == 0.25
    assert back.index[0] == pd.Timestamp('2026-01-12 00:00')


def test_reads_a_real_download_with_bom_and_crlf(tmp_path: Path):
    gen, _ = synthetic_week()
    path = tmp_path / 'download.csv'
    path.write_text(to_smard_csv(gen), encoding='utf-8-sig', newline='\r\n')
    back = read_smard(path)
    assert list(back.columns) == GENERATION_COLUMNS
    assert np.allclose(back.drop(columns='Nuclear'), gen.drop(columns='Nuclear').round(2))


def test_reads_the_german_export_with_english_names(tmp_path: Path):
    path = tmp_path / 'verbrauch.csv'
    path.write_bytes(GERMAN_EXPORT.encode('utf-8-sig'))
    table = read_smard(path)
    assert list(table.columns) == CONSUMPTION_COLUMNS
    assert table.index.tolist() == [pd.Timestamp('2026-01-12 00:00'), pd.Timestamp('2026-01-12 01:00')]
    assert table[LOAD].tolist() == [53862.40, 52156.79]
    assert np.isnan(table['Hydro pumped storage'].iloc[1]) and table['Residual load'].iloc[1] == 28228.64


def test_rejects_a_file_that_is_not_a_smard_export(tmp_path: Path):
    path = tmp_path / 'other.csv'
    path.write_text('time;value\n1;2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not a SMARD export'):
        read_smard(path)


def test_samples_parse_to_hourly_frames():
    gen, cons = read_smard(GENERATION_CSV), read_smard(CONSUMPTION_CSV)
    assert len(gen) == len(cons) == HOURS
    assert gen.index.freq is None and (gen.index[1] - gen.index[0]) == pd.Timedelta(hours=1)
    assert gen['Nuclear'].isna().all()
    assert np.allclose(residual_load(gen, cons), cons['Residual load'], atol=0.02)


def test_duration_curve_is_sorted_and_keeps_every_hour():
    _, cons = synthetic_week()
    curve = duration_curve(cons[LOAD])
    assert len(curve) == HOURS and (np.diff(curve.to_numpy()) <= 0).all()
    assert curve.index[0] == 1 and curve.index[-1] == HOURS


def test_critical_hours_sit_on_the_wind_front():
    gen, cons = synthetic_week()
    crit = critical_hours(gen, cons)
    assert 0 < len(crit) < HOURS / 4
    assert crit.index.min() >= pd.Timestamp('2026-01-14') and crit.index.max() < pd.Timestamp('2026-01-17')


def test_daily_summary_has_seven_days_and_sane_shares():
    gen, cons = synthetic_week()
    daily = daily_summary(gen, cons)
    assert list(daily['weekday']) == ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    assert daily['share_pct'].between(0, 100).all()
    assert daily['share_pct'].idxmax() == pd.Timestamp('2026-01-15')
