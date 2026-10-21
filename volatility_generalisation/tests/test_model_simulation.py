import numpy as np

import volatility_generalisation.modelling_code.model_simulation as model_simulation

from volatility_generalisation.modelling_code.model_simulation import (
    dual_lr_single_value,
    leaky_beta_asymmetric_single_value,
    leaky_beta_single_value,
    pearce_hall_single_value,
    rescorla_wagner_single_value,
    run_model_scan,
)


def test_rescorla_wagner_single_value_derives_complement():
    choice_prob, value_0, value_1, pe = rescorla_wagner_single_value(
        outcome_option0=1.0,
        trial_number=0,
        outcome_option1=0.0,
        value_0=0.5,
        value_1=0.5,
        prev_pos=0.5,
        learning_rate=0.3,
        temperature=0.5,
    )

    assert np.isclose(choice_prob, 0.5)
    assert np.isclose(value_0, 0.65)
    assert np.isclose(value_1, 0.35)
    assert np.isclose(pe, 0.5)


def test_dual_lr_single_value_derives_complement():
    choice_prob, value_0, value_1, pe = dual_lr_single_value(
        outcome_option0=1.0,
        trial_number=0,
        outcome_option1=0.0,
        value_0=0.5,
        value_1=0.5,
        prev_pos=0.5,
        learning_rate_pos=0.3,
        learning_rate_neg=0.1,
        temperature=0.5,
    )

    assert np.isclose(choice_prob, 0.5)
    assert np.isclose(value_0, 0.65)
    assert np.isclose(value_1, 0.35)
    assert np.isclose(pe, 0.5)


def test_pearce_hall_single_value_uses_single_prediction_error():
    choice_prob, value_0, value_1, assoc_0, pe = pearce_hall_single_value(
        outcome_option0=1.0,
        trial_number=0,
        outcome_option1=0.0,
        value_0=0.5,
        value_1=0.5,
        assoc_0=0.5,
        prev_pos=0.5,
        kappa=0.5,
        gamma=0.2,
        temperature=0.5,
    )

    assert np.isclose(choice_prob, 0.5)
    assert np.isclose(value_0, 0.625)
    assert np.isclose(value_1, 0.375)
    assert np.isclose(assoc_0, 0.5)
    assert np.isclose(pe, 0.5)


def test_leaky_beta_single_value_derives_complement():
    (
        choice_prob,
        beta_a_0,
        beta_b_0,
        beta_a_1,
        beta_b_1,
        value_0,
        value_1,
        variance_0,
        variance_1,
    ) = leaky_beta_single_value(
        outcome_option0=1.0,
        trial_number=0,
        outcome_option1=0.0,
        beta_a_0=1.0,
        beta_b_0=1.0,
        prev_pos=0.5,
        tau=1.0,
        decay=0.9,
        temperature=0.5,
    )

    assert np.isclose(choice_prob, 0.5)
    assert np.isclose(beta_a_0, 2.0)
    assert np.isclose(beta_b_0, 1.0)
    assert np.isclose(beta_a_1, 1.0)
    assert np.isclose(beta_b_1, 2.0)
    assert np.isclose(value_0, 2.0 / 3.0)
    assert np.isclose(value_1, 1.0 / 3.0)
    assert np.isclose(variance_0, 1.0 / 18.0)
    assert np.isclose(variance_1, 1.0 / 18.0)


def test_leaky_beta_asymmetric_single_value_derives_complement():
    (
        choice_prob,
        beta_a_0,
        beta_b_0,
        beta_a_1,
        beta_b_1,
        value_0,
        value_1,
        variance_0,
        variance_1,
    ) = leaky_beta_asymmetric_single_value(
        outcome_option0=1.0,
        trial_number=0,
        outcome_option1=0.0,
        beta_a_0=1.0,
        beta_b_0=1.0,
        prev_pos=0.5,
        tau_pos=2.0,
        tau_neg=3.0,
        decay=0.9,
        temperature=0.5,
    )

    assert np.isclose(choice_prob, 0.5)
    assert np.isclose(beta_a_0, 3.0)
    assert np.isclose(beta_b_0, 1.0)
    assert np.isclose(beta_a_1, 1.0)
    assert np.isclose(beta_b_1, 3.0)
    assert np.isclose(value_0, 0.75)
    assert np.isclose(value_1, 0.25)
    assert np.isclose(variance_0, 3.0 / 80.0)
    assert np.isclose(variance_1, 3.0 / 80.0)


def test_get_model_info_accepts_retained_models():
    from volatility_generalisation.modelling_code.model_simulation import get_model_info

    for model_name in [
        "rescorla_wagner_single_value",
        "dual_lr_single_value",
        "pearce_hall_single_value",
        "leaky_beta_single_value",
        "leaky_beta_asymmetric_single_value",
    ]:
        info = get_model_info(model_name)
        assert info is not None


def test_get_model_info_rejects_removed_models():
    from volatility_generalisation.modelling_code.model_simulation import get_model_info

    for model_name in [
        "rescorla_wagner",
        "rescorla_wagner_independent",
        "dual_lr",
        "dual_lr_independent",
        "pearce_hall",
        "pearce_hall_single_value_dual_assoc",
        "pearce_hall_independent",
        "leaky_beta",
        "leaky_beta_independent",
        "leaky_beta_asymmetric",
        "leaky_beta_asymmetric_independent",
        "leaky_beta_asymmetric_single_value_reparam",
    ]:
        try:
            get_model_info(model_name)
        except ValueError:
            continue
        raise AssertionError(f"Expected {model_name} to be rejected")


def test_independent_model_callables_are_absent_from_module_namespace():
    for name in [
        "rescorla_wagner_independent",
        "dual_lr_independent",
        "pearce_hall_independent",
        "leaky_beta_independent",
        "leaky_beta_asymmetric",
        "leaky_beta_asymmetric_independent",
    ]:
        assert not hasattr(model_simulation, name)


def test_run_model_scan_works_for_retained_models():
    trial_data = {
        "outcomes_0": np.array([1.0]),
        "outcomes_1": np.array([0.0]),
        "choices": np.array([1.0]),
    }

    model_params = {
        "rescorla_wagner_single_value": {"learning_rate": 0.3, "temperature": 0.5},
        "dual_lr_single_value": {"learning_rate_pos": 0.3, "learning_rate_neg": 0.1, "temperature": 0.5},
        "pearce_hall_single_value": {"kappa": 0.5, "gamma": 0.2, "temperature": 0.5},
        "leaky_beta_single_value": {"tau": 1.0, "decay": 0.9, "temperature": 0.5},
        "leaky_beta_asymmetric_single_value": {"tau_pos": 2.0, "tau_neg": 3.0, "decay": 0.9, "temperature": 0.5},
    }

    expected_keys = {
        "rescorla_wagner_single_value": ["prev_pos", "prediction_error"],
        "dual_lr_single_value": ["prev_pos", "prediction_error"],
        "pearce_hall_single_value": ["prev_pos", "value_0", "value_1", "assoc_0", "pe"],
        "leaky_beta_single_value": ["prev_pos", "value_0", "value_1", "variance_0", "variance_1"],
        "leaky_beta_asymmetric_single_value": ["prev_pos", "value_0", "value_1", "variance_0", "variance_1"],
    }

    for model_name, params in model_params.items():
        _, outputs = run_model_scan(model_name, params, trial_data)
        assert set(outputs.keys()) == set(expected_keys[model_name])
        for key in expected_keys[model_name]:
            assert outputs[key].shape[0] == 1
