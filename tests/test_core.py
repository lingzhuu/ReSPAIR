import numpy as np

import respair
from respair import core


def test_package_api():
    assert callable(respair.fit)
    assert isinstance(respair.__version__, str)


def test_pearson_residuals_shape_and_clip():
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, (100, 30))
    r = core.pearson_residuals(counts)
    assert r.shape == counts.shape
    assert np.all(np.abs(r) <= np.sqrt(100) + 1e-12)


def test_spatial_basis_orthonormal():
    xx, yy = np.meshgrid(np.arange(10), np.arange(10))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    evals, U = core.spatial_basis(coords, k=4)
    assert U.shape == (100, 99)
    assert np.all(np.diff(evals) >= -1e-10)  # low to high frequency
    np.testing.assert_allclose(U.T @ U, np.eye(99), atol=1e-8)


def test_bh_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej = core.bh(p, alpha=0.05)
    assert rej.tolist() == [True, True] + [False] * 8


def test_bh_none_rejected():
    assert not core.bh(np.array([0.5, 0.9])).any()


def test_lambda_gc_null_is_one():
    z = np.random.default_rng(0).standard_normal(200_000)
    assert abs(core.lambda_gc(z) - 1) < 0.02


def test_residualize_removes_covariates():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((200, 3))
    X = C @ rng.standard_normal((3, 5)) + 0.1 * rng.standard_normal((200, 5))
    R = core._residualize(X, C)
    np.testing.assert_allclose(C.T @ R, 0, atol=1e-8)
