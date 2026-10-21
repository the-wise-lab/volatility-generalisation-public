"""Model fitting and data preparation for the volatility generalisation study.

The package imports the data preparation functions directly. Import the fitting
and simulation functions from their own modules.
"""

__version__ = "0.1.0"
__author__ = "Phoenix Pavlik, Tin Nok Isaac Choy, Toby Wise"

# Import data preparation utilities for easy access
from .modelling_code.data_preparation import (
    prepare_complementary_data,
    prepare_data_for_fitting,
)

# This file imports only the data preparation functions above. The model
# fitting and simulation functions are not imported here, so import them from
# their own modules when you need them, for example:
# from volatility_generalisation.modelling_code.hierarchical_bayes import fit_mcmc
# from volatility_generalisation.modelling_code.model_simulation import run_model_scan

__all__ = [
    # Data preparation
    "prepare_complementary_data",
    "prepare_data_for_fitting",
]
