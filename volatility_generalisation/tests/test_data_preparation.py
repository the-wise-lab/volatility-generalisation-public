"""
Tests for data preparation utilities.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from volatility_generalisation.modelling_code.data_preparation import (
    prepare_complementary_data,
    prepare_data_for_fitting,
)


def load_prepare_data_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _condition_rows(subject_id, n_trials, first_trial=1):
    """Build one participant's rows for ``prepare_and_save_condition``.

    Args:
        subject_id: Participant ID.
        n_trials: Number of trials.
        first_trial: Trial number of the first row.

    Returns:
        DataFrame with ``subject_id``, ``choice``, ``trial_num`` and
        ``outcomes_0`` columns.
    """
    trials = np.arange(n_trials)
    return pd.DataFrame({
        'subject_id': subject_id,
        'choice': 1.0 + (trials % 2),
        'trial_num': trials + first_trial,
        'outcomes_0': (trials % 3 == 0).astype(int),
    })


def _raw_block_rows(subject_id, task_version, block, repeat_final=False):
    """Build one participant's raw CSV rows for one block.

    Block 1 is trials 0-119 and block 2 trials 120-239. With
    ``repeat_final``, block 2 gets trial 240, repeating trial 239's outcome and
    ball colour but with a different choice (as in 0.5% of the real repeats).
    Choices come in runs of 10 so the low-engagement check passes.

    Args:
        subject_id: Participant ID.
        task_version: Task version (1-4).
        block: Block number (1 or 2).
        repeat_final: Whether to add the repeated final log row.

    Returns:
        DataFrame of raw trial rows.
    """
    trials = np.arange(120)
    rows = pd.DataFrame({
        'subject_id': subject_id,
        'task_version': task_version,
        'block': block,
        'trial_num': trials + 120 * (block - 1),
        'choice': 1.0 + (trials // 10) % 2,
        'exploded': (trials % 4 == 0).astype(int),
        'ball_color': np.where(trials % 2 == 0, 'red', 'blue'),
    })
    if repeat_final:
        repeat = rows.iloc[[-1]].copy()
        repeat['trial_num'] += 1
        repeat['choice'] = 3.0 - repeat['choice']
        rows = pd.concat([rows, repeat], ignore_index=True)
    return rows


def _raw_participant(subject_id, task_version=1):
    """Build both blocks for one participant, with the repeated final row.

    Args:
        subject_id: Participant ID.
        task_version: Task version (1-4).

    Returns:
        DataFrame of the participant's raw rows (120 + 121).
    """
    return pd.concat([
        _raw_block_rows(subject_id, task_version, 1),
        _raw_block_rows(subject_id, task_version, 2, repeat_final=True),
    ], ignore_index=True)


def test_prepare_complementary_data_covers_all_choice_outcome_cases():
    """Each choice/outcome pairing gives the expected arrays, and the input is unchanged."""
    # One participant, four trials: every combination of the option chosen
    # (1.0 = first option, 2.0 = second option) and the outcome it gave.
    df = pd.DataFrame({
        'subject_id': ['subj_01'] * 4,
        'choice': [1.0, 1.0, 2.0, 2.0],
        'trial_num': [0, 1, 2, 3],
        'outcomes_0': [1, 0, 1, 0],
    })
    df_before = df.copy()

    data = prepare_complementary_data(df)

    # Trial 0: chose the first option and it paid out, so the other option
    #          did not (outcomes 1 / 0, choice recorded as 1).
    # Trial 1: chose the first option and it did not pay out, so the other
    #          option did (outcomes 0 / 1, choice 1).
    # Trial 2: chose the second option and it paid out (outcomes 0 / 1,
    #          choice recorded as 0).
    # Trial 3: chose the second option and it did not pay out (outcomes 1 / 0,
    #          choice 0).
    np.testing.assert_array_equal(data['outcomes_0'], [[1.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(data['outcomes_1'], [[0.0, 1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(data['choices'], [[1.0, 1.0, 0.0, 0.0]])
    assert data['outcomes_0'].shape == (1, 4)
    assert data['outcomes_1'].shape == (1, 4)
    assert data['choices'].shape == (1, 4)
    assert data['subject_ids'].tolist() == ['subj_01']

    # Preparing the data must not change the participant's input rows.
    pd.testing.assert_frame_equal(df, df_before)


def test_prepare_complementary_multiple_subjects():
    """Test preparation with multiple subjects."""
    df = pd.DataFrame({
        'subject_id': ['subj_01', 'subj_01', 'subj_02', 'subj_02'],
        'choice': [1.0, 2.0, 1.0, 1.0],
        'trial_num': [0, 1, 0, 1],
        'outcomes_0': [1, 0, 0, 1]
    })
    
    data = prepare_complementary_data(df)
    
    # Check shapes
    assert data['outcomes_0'].shape == (2, 2)
    assert data['outcomes_1'].shape == (2, 2)
    assert data['choices'].shape == (2, 2)
    assert len(data['subject_ids']) == 2


def test_prepare_data_for_fitting_jax():
    """Test that JAX conversion works."""
    df = pd.DataFrame({
        'subject_id': ['subj_01'] * 3,
        'choice': [1.0, 2.0, 1.0],
        'trial_num': [0, 1, 2],
        'outcomes_0': [1, 0, 1]
    })
    
    # With JAX
    data_jax = prepare_data_for_fitting(df, use_jax=True)
    
    # Check that we got JAX arrays
    import jax.numpy as jnp
    assert isinstance(data_jax['outcomes_0'], jnp.ndarray)
    assert isinstance(data_jax['outcomes_1'], jnp.ndarray)
    assert isinstance(data_jax['choices'], jnp.ndarray)
    
    # Without JAX
    data_numpy = prepare_data_for_fitting(df, use_jax=False)
    assert isinstance(data_numpy['outcomes_0'], np.ndarray)


def test_exclusion_table_records_every_criterion_each_participant_met():
    """Every criterion is checked for everyone; meeting one never hides another."""
    prepare_data = load_prepare_data_module()
    trials = np.arange(120)
    engaged = 1.0 + (trials // 10) % 2  # runs of 10, passes both choice checks

    def rows(subject_id, choices):
        return pd.DataFrame({'subject_id': subject_id, 'choice': choices})

    condition_data = {
        # 'flagged' always chooses option 1, has 119 trials, and is on the
        # manual list; 'one_block' has no StableV1 trials at all.
        'VolatileV1': pd.concat([
            rows('fine', engaged), rows('flagged', np.ones(119)),
            rows('one_block', engaged),
        ], ignore_index=True),
        'StableV1': pd.concat([
            rows('fine', engaged), rows('flagged', engaged),
        ], ignore_index=True),
    }

    table = prepare_data.assess_exclusion_criteria(
        condition_data, manual_exclusions=['flagged', 'not_in_data'],
        input_path='synthetic.csv',
    )

    assert table.columns.tolist() == prepare_data.QC_AUDIT_COLUMNS
    met = table.set_index(['condition', 'subject_id'])[
        prepare_data.EXCLUSION_CRITERIA + ['excluded']
    ]
    # Order: manual list, low choice variability, extreme mean choice, low
    # engagement in another condition, incomplete trials, then excluded.
    expected = {
        ('VolatileV1', 'fine'): [False, False, False, False, False, False],
        ('VolatileV1', 'flagged'): [True, True, True, False, True, True],
        ('VolatileV1', 'one_block'): [False, False, False, False, False, False],
        ('StableV1', 'fine'): [False, False, False, False, False, False],
        ('StableV1', 'flagged'): [True, False, False, True, False, True],
        ('StableV1', 'one_block'): [False, False, False, False, True, True],
    }
    assert {key: met.loc[key].tolist() for key in expected} == expected
    assert len(table) == len(expected)
    reasons = table.set_index(['condition', 'subject_id'])['exclusion_reasons']
    assert reasons[('VolatileV1', 'flagged')] == (
        'on_manual_exclusion_list; low_choice_variability; extreme_mean_choice; '
        'incomplete_trials'
    )
    assert reasons[('VolatileV1', 'fine')] == ''
    assert table.set_index(['condition', 'subject_id']).loc[
        ('StableV1', 'one_block'), 'observed_trials'
    ] == 0


if __name__ == "__main__":
    # Run tests
    test_prepare_complementary_data_covers_all_choice_outcome_cases()
    test_prepare_complementary_multiple_subjects()
    test_prepare_data_for_fitting_jax()
    print("All tests passed!")


def test_drop_repeated_final_trial_removes_only_the_repeat():
    """The repeated block-2 row is dropped; every other row is kept unchanged."""
    prepare_data = load_prepare_data_module()
    data = pd.concat([_raw_participant('p1'), _raw_participant('p2')], ignore_index=True)

    result = prepare_data.drop_repeated_final_trial(data)

    assert result.groupby(['subject_id', 'block']).size().tolist() == [120, 120, 120, 120]
    assert 240 not in result['trial_num'].tolist()
    expected = data[data['trial_num'] != 240]
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("change, message", [
    ("block_1_extra", "Extra trial outside block 2"),
    ("different_outcome", "does not repeat the trial before it"),
    ("different_colour", "does not repeat the trial before it"),
    ("two_extra", "Too many trials"),
    ("no_ball_color", "missing columns \\['ball_color'\\]"),
])
def test_drop_repeated_final_trial_rejects_other_extra_rows(change, message):
    """Extra rows that are not the documented repeat stop the run."""
    prepare_data = load_prepare_data_module()
    block_1 = _raw_block_rows('p1', 1, 1)
    block_2 = _raw_block_rows('p1', 1, 2, repeat_final=True)
    if change == "block_1_extra":
        block_1 = _raw_block_rows('p1', 1, 1, repeat_final=True)
        block_2 = _raw_block_rows('p1', 1, 2)
    elif change == "different_outcome":
        block_2.loc[block_2.index[-1], 'exploded'] = 1 - block_2['exploded'].iloc[-1]
    elif change == "different_colour":
        block_2.loc[block_2.index[-1], 'ball_color'] = 'green'
    elif change == "two_extra":
        extra = block_2.iloc[[-1]].copy()
        extra['trial_num'] += 1
        block_2 = pd.concat([block_2, extra], ignore_index=True)
    data = pd.concat([block_1, block_2], ignore_index=True)
    if change == "no_ball_color":
        data = data.drop(columns='ball_color')

    with pytest.raises(ValueError, match=message):
        prepare_data.drop_repeated_final_trial(data)


@pytest.mark.parametrize("counts, observed", [((120, 121), 121), ((119, 119), 119)])
def test_exclusion_check_rejects_wrong_maximum_trial_count(counts, observed):
    """The most trials any participant has must equal the design's 120."""
    prepare_data = load_prepare_data_module()
    condition_data = pd.concat([
        _condition_rows('a', counts[0]), _condition_rows('b', counts[1]),
    ], ignore_index=True)

    with pytest.raises(ValueError, match=f"'VolatileV1'.*{observed}.*has 120"):
        prepare_data.assess_exclusion_criteria({'VolatileV1': condition_data}, [])


@pytest.mark.parametrize("last_trial_num", [121, 119])
def test_prepare_condition_rejects_gaps_or_repeats_in_trial_numbers(tmp_path, last_trial_num):
    """120 trials numbered with a gap (or a repeat) would leave an invented trial."""
    prepare_data = load_prepare_data_module()
    gappy = _condition_rows('gappy', 120)
    gappy.loc[gappy.index[-1], 'trial_num'] = last_trial_num
    condition_data = pd.concat([_condition_rows('fine', 120), gappy], ignore_index=True)

    with pytest.raises(ValueError, match="not consecutive for participants \\['gappy'\\]"):
        prepare_data.prepare_and_save_condition('TestCondition', condition_data, [], tmp_path)


def test_main_saves_only_participants_the_exclusion_table_keeps(tmp_path, monkeypatch):
    """End to end: the repeat is dropped and the saved files match the table.

    Two participants per task version. new_RLT_v1_1 always chooses option 1 in
    block 1 (low engagement there, so removed from both version 1 files) and
    new_RLT_v2_0 is on the manual exclusion list.
    """
    import sys

    prepare_data = load_prepare_data_module()
    raw = pd.concat([
        _raw_participant(f'new_RLT_v{version}_{i}', task_version=version)
        for version in (1, 2, 3, 4) for i in range(2)
    ], ignore_index=True)
    raw.loc[(raw['subject_id'] == 'new_RLT_v1_1') & (raw['block'] == 1), 'choice'] = 1.0
    csv_path = tmp_path / 'raw.csv'
    raw.to_csv(csv_path, index=False)
    output_dir = tmp_path / 'prepared'
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[data]\nexcluded_subjects = ["new_RLT_v2_0"]\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data.py",
            "--config",
            str(config_path),
            "--main-data",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    prepare_data.main()

    table = pd.read_csv(output_dir / prepare_data.QC_AUDIT_FILENAME)
    assert len(table) == 16  # 8 participants x 2 blocks each
    files = sorted(output_dir.glob('*.npz'))
    assert len(files) == 8
    for path in files:
        kept = table.loc[(table['condition'] == path.stem) & ~table['excluded'], 'subject_id']
        with np.load(path, allow_pickle=True) as saved:
            assert saved['subject_ids'].tolist() == kept.tolist(), path.stem
            assert saved['choices'].shape == (len(kept), 120), path.stem
    excluded = table[table['excluded']]
    assert sorted(zip(excluded['condition'], excluded['subject_id'],
                      excluded['exclusion_reasons'])) == [
        ('StableV1', 'new_RLT_v1_1', 'low_engagement_in_another_condition'),
        ('StableV2', 'new_RLT_v2_0', 'on_manual_exclusion_list'),
        ('VolatileV1', 'new_RLT_v1_1', 'low_choice_variability; extreme_mean_choice'),
        ('VolatileV2', 'new_RLT_v2_0', 'on_manual_exclusion_list'),
    ]
    # Counts per condition, then each participant once across conditions:
    # participants, excluded, kept, then the five criteria in order.
    counts = pd.read_csv(output_dir / prepare_data.QC_COUNTS_FILENAME).set_index('condition')
    assert counts.loc['VolatileV1'].tolist() == [2, 1, 1, 0, 1, 1, 0, 0]
    assert counts.loc['StableV1'].tolist() == [2, 1, 1, 0, 0, 0, 1, 0]
    assert counts.loc['StableV4'].tolist() == [2, 0, 2, 0, 0, 0, 0, 0]
    assert counts.loc['any_condition'].tolist() == [8, 2, 6, 1, 1, 1, 1, 0]
    assert len(counts) == 9


@pytest.mark.parametrize(
    "config_text, error_type",
    [
        (None, FileNotFoundError),
        ("[data]\nrolling_window = 10\nrolling_window = 5\n", ValueError),
    ],
)
def test_main_stops_before_writing_when_config_is_missing_or_broken(
    tmp_path, monkeypatch, config_text, error_type
):
    """A missing or unreadable config stops main before any output is written."""
    import sys

    prepare_data = load_prepare_data_module()
    output_dir = tmp_path / "prepared"
    config_path = tmp_path / "config.toml"
    if config_text is not None:
        config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data.py",
            "--config",
            str(config_path),
            "--main-data",
            str(tmp_path / "raw.csv"),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(error_type, match="config.toml"):
        prepare_data.main()

    assert not output_dir.exists()
