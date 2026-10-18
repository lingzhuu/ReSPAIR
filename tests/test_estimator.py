import numpy as np
import pytest

from respair import estimator as full


def test_gp_basis_unit_prior_variance():
    xx, yy = np.meshgrid(np.arange(15.0), np.arange(15.0))
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    Phi = full.GPBasis(coords, 3.0)(coords)
    # F = Φ α with α ~ N(0, I) has prior variance diag(ΦΦᵀ) ≈ 1 inside the inducing grid
    assert np.median((Phi ** 2).sum(1)) == pytest.approx(1.0, abs=0.05)


def test_kernel_map_same_grid_is_identity():
    c = np.random.default_rng(0).random((30, 2))
    assert full.kernel_map(c, c) is None
    A = full.kernel_map(c, c + 0.01)
    np.testing.assert_allclose(A.sum(1), 1)


def test_pln_estep_matches_numerical_mode():
    rng = np.random.default_rng(0)
    Y = rng.poisson(3.0, (50, 4)).astype(float)
    prior = np.full_like(Y, np.log(3.0))
    s2 = np.array([0.1, 0.5, 1.0, 2.0])
    x, v = full._pln_estep(Y, prior, s2, np.zeros_like(Y))
    grad = Y - np.exp(x) - (x - prior) / s2
    np.testing.assert_allclose(grad, 0, atol=1e-6)
    assert np.all(v > 0)


def test_two_groups_pip_is_monotone_without_prior():
    z = np.random.default_rng(0).standard_normal(5000)
    z[:50] += 5
    pip, _ = full.two_groups_pip(z)
    order = np.argsort(np.abs(z))
    assert np.all(np.diff(pip[order]) >= -1e-12)


def test_fit_outputs_and_power(small_sim):
    d = small_sim
    res = full.fit(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"], n_iter=4, n_draws=2)
    p, q = d["truth"].shape
    assert res["z"].shape == (p, q) and res["pip"].shape == (p, q)
    assert res["W"].shape == (q, p)
    assert res["F_g"].shape[1] == res["K"] == len(res["lengths"])
    null = ~d["truth"]
    assert (res["p"][null] < 0.05).mean() < 0.08
    assert (res["p"][d["truth"]] < 0.05).mean() > 0.6


def test_conditional_test_separates_isozyme():
    """A gene correlated with the true enzyme is not significant once the enzyme is in W."""
    rng = np.random.default_rng(0)
    L, p = 400, 3
    X = rng.standard_normal((L, p))
    X[:, 1] = 0.9 * X[:, 0] + np.sqrt(1 - 0.81) * X[:, 1]
    R = (0.5 * X[:, 0] + rng.standard_normal(L))[:, None]
    W = np.zeros((1, p)); W[0, 0] = 0.5
    T, V = full.conditional_test(X, R, W, band_size=10)
    z = T[:, 0] / np.sqrt(V[:, 0])
    assert abs(z[0]) > 5
    assert abs(z[1]) < 3
