from conductor import capacitor_energy, inductor_energy, joule_losses, line_losses, resistance


def test_exercise_1_resistance_and_losses():
    assert abs(resistance() - 5.64) < 1e-9
    assert abs(joule_losses(1000.0) - 5.64e6) < 1e-3


def test_exercise_2_losses_scale_as_one_over_v_squared():
    i200, p200 = line_losses(200e6, 200e3)
    i400, p400 = line_losses(200e6, 400e3)
    assert (i200, i400) == (1000.0, 500.0)
    assert abs(p200 - 11.28e6) < 1e-3 and abs(p400 - 2.82e6) < 1e-3
    assert abs(p200 / p400 - 4.0) < 1e-12


def test_exercise_4_stored_energy():
    assert inductor_energy(1.0, 1000.0) == 500e3
    assert abs(capacitor_energy(1e-6, 230.9e3) - 26.66e3) < 5
