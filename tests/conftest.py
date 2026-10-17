import pytest

from respair.simulate import simulate


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
    """小规模模拟（20×20 网格），几秒内跑完。"""
    return simulate(seed=1, side=20, p=300, q=60, n_conf_genes=60, n_conf_mets=15,
                    n_links=40, link_strength=1.0, n_neg=100)
