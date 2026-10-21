from chessrl.levels.tune import acpl, fit_error_scale, prepare, sample_positions, target_acpl


def test_target_curve_is_decreasing():
    vals = [target_acpl(e) for e in (600, 1000, 1400, 1800, 2200)]
    assert vals == sorted(vals, reverse=True)


def test_error_scale_controls_acpl_and_fit_hits_target():
    positions = prepare(sample_positions(10, seed=3), [2], ref_depth=2, seconds=0.5)
    low = acpl(positions, 2, 1200, 0.05, repeats=3)
    high = acpl(positions, 2, 1200, 4.0, repeats=3)
    assert high > low
    scale, got = fit_error_scale(positions, 2, 1200)
    assert 0.02 <= scale <= 6.0
    assert abs(got - target_acpl(1200)) < 0.35 * target_acpl(1200) or scale in (0.02, 6.0)
