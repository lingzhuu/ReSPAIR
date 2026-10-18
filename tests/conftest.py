import os
import sys

import pytest

# the simulator is benchmark code, not part of the package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "scripts"))
from simulate import simulate  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run full-size simulations")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def small_sim():
    """Small simulation (20×20 grid) that runs in seconds."""
    return simulate(seed=1, side=20, p=300, q=60, n_conf_genes=60, n_conf_mets=15,
                    n_links=40, link_strength=1.0, n_neg=100)
