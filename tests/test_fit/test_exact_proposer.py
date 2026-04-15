"""Tests for the exact proposer with expanded KL regularization."""

import numpy as np

from olmix.fit.utils import LogLinearExactProposer, build_expansion_matrix, expand_collapsed_weights


class DummyPredictor:
    def __init__(self, model):
        self.model = np.asarray(model, dtype=float)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


class TestExpandedKLHelpers:
    def test_build_expansion_matrix(self):
        matrix, expanded_keys = build_expansion_matrix(
            {"existing": 0.7, "new": 0.3},
            expanded_prior={"existing:a": 0.2, "existing:b": 0.5, "new": 0.3},
            source_mixtures={"existing": {"existing:a": 0.9, "existing:b": 0.1}},
        )

        np.testing.assert_allclose(matrix, np.array([[0.9, 0.0], [0.1, 0.0], [0.0, 1.0]]))
        assert expanded_keys == ["existing:a", "existing:b", "new"]

    def test_expand_collapsed_weights_uses_frozen_source_mixture(self):
        expanded = expand_collapsed_weights(
            {"existing": 0.6, "new": 0.4},
            original_prior={"existing:a": 0.2, "existing:b": 0.5, "new": 0.3},
            collapsed_prior={"existing": 0.7, "new": 0.3},
            source_mixtures={"existing": {"existing:a": 0.9, "existing:b": 0.1}},
        )

        assert expanded == {"existing:a": 0.54, "existing:b": 0.06, "new": 0.4}


class TestLogLinearExactProposer:
    def test_expanded_kl_matches_bruteforce_scalar_optimum(self):
        proposer = LogLinearExactProposer()
        predictor = [DummyPredictor([0.0, 0.1, 0.0])]

        prior = {"existing": 0.7, "new": 0.3}
        expanded_prior = {"existing:a": 0.2, "existing:b": 0.5, "new": 0.3}
        source_mixtures = {"existing": {"existing:a": 0.9, "existing:b": 0.1}}

        weights = proposer.propose(
            predictor=predictor,
            prior_distributions=prior,
            expanded_prior_distributions=expanded_prior,
            expanded_source_mixtures=source_mixtures,
            token_counts={"existing": 700, "new": 300},
            kl_reg=0.2,
        )

        alpha = float(weights[1])

        grid = np.linspace(0.0, 1.0, 4001)
        expanded_q = np.array(list(expanded_prior.values()), dtype=float)
        expanded_q /= expanded_q.sum()
        collapsed_q = np.array(list(prior.values()), dtype=float)
        collapsed_q /= collapsed_q.sum()
        old_mix = np.array([0.9, 0.1, 0.0], dtype=float)
        new_mix = np.array([0.0, 0.0, 1.0], dtype=float)

        expanded_values = []
        collapsed_values = []
        for candidate in grid:
            collapsed_x = np.array([1.0 - candidate, candidate], dtype=float)
            expanded_x = (1.0 - candidate) * old_mix + candidate * new_mix
            loss = float(np.exp(0.1 * collapsed_x[0]))
            expanded_values.append(loss + 0.2 * _kl(expanded_x, expanded_q))
            collapsed_values.append(loss + 0.2 * _kl(collapsed_x, collapsed_q))

        expanded_alpha = float(grid[int(np.argmin(expanded_values))])
        collapsed_alpha = float(grid[int(np.argmin(collapsed_values))])

        assert abs(alpha - expanded_alpha) < 1e-2
        assert alpha > collapsed_alpha + 0.05

    def test_without_expanded_kl_uses_collapsed_solution(self):
        proposer = LogLinearExactProposer()
        predictor = [DummyPredictor([0.0, 0.1, 0.0])]
        prior = {"existing": 0.7, "new": 0.3}

        weights = proposer.propose(
            predictor=predictor,
            prior_distributions=prior,
            token_counts={"existing": 700, "new": 300},
            kl_reg=0.2,
        )

        alpha = float(weights[1])

        grid = np.linspace(0.0, 1.0, 4001)
        collapsed_q = np.array(list(prior.values()), dtype=float)
        collapsed_q /= collapsed_q.sum()
        objective = []
        for candidate in grid:
            collapsed_x = np.array([1.0 - candidate, candidate], dtype=float)
            loss = float(np.exp(0.1 * collapsed_x[0]))
            objective.append(loss + 0.2 * _kl(collapsed_x, collapsed_q))

        expected_alpha = float(grid[int(np.argmin(objective))])
        assert abs(alpha - expected_alpha) < 1e-2
