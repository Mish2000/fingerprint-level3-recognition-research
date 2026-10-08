import numpy as np
import pytest

from fpl3.eval import metrics, verification as v


def score_set(genuine, impostor, n_subjects=4):
    rng = np.random.default_rng(0)
    return v.Scores(
        genuine=np.asarray(genuine, float),
        genuine_subject=rng.integers(0, n_subjects, len(genuine)),
        impostor=np.asarray(impostor, float),
        impostor_probe=rng.integers(0, n_subjects, len(impostor)),
        impostor_reference=rng.integers(0, n_subjects, len(impostor)),
        n_subjects=n_subjects,
    )


def test_unit_weights_reproduce_the_plain_tar_at_far_with_ties_and_failures():
    rng = np.random.default_rng(1)
    genuine = list(rng.integers(0, 60, 300)) + [None] * 5
    impostor = list(rng.integers(0, 30, 5000)) + [None] * 20
    s = score_set([v.FAILED if x is None else x for x in genuine], [v.FAILED if x is None else x for x in impostor])
    r = v.rates(s, v.Grid(s))
    for far in (0.1, 0.01, 0.001, 0.0001, 0.0):
        expected = metrics.tar_at_far(genuine, impostor, far)
        got = v.tar_at_far(r, far)
        assert got["tar"] == pytest.approx(expected["tar"]) and got["far"] == pytest.approx(expected["far"])


def test_eer_where_far_meets_frr():
    s = score_set([2, 3, 4, 5], [0, 1, 2, 3])
    assert v.eer(v.rates(s, v.Grid(s))) == pytest.approx(0.25)


def test_a_subject_drawn_twice_counts_like_its_pairs_twice():
    s = score_set([5, 6, 7, 1], [0, 1, 2, 3, 6, 2])
    weights = np.array([2.0, 1.0, 0.0, 1.0])
    gen_w = weights[s.genuine_subject].astype(int)
    imp_w = (weights[s.impostor_probe] * weights[s.impostor_reference]).astype(int)
    copied = v.Scores(np.repeat(s.genuine, gen_w), np.zeros(gen_w.sum(), int),
                      np.repeat(s.impostor, imp_w), np.zeros(imp_w.sum(), int), np.zeros(imp_w.sum(), int), 1)
    weighted, plain = v.rates(s, v.Grid(s), weights), v.rates(copied, v.Grid(copied))
    for far in (0.5, 0.2, 0.0):
        assert v.tar_at_far(weighted, far)["tar"] == pytest.approx(v.tar_at_far(plain, far)["tar"])
    assert v.eer(weighted) == pytest.approx(v.eer(plain))


def test_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(2)
    a = score_set(rng.normal(3, 1, 200), rng.normal(0, 1, 2000), n_subjects=20)
    b = v.Scores(a.genuine + rng.normal(0, 0.5, 200), a.genuine_subject, a.impostor, a.impostor_probe,
                 a.impostor_reference, 20)
    first = v.subject_bootstrap({"a": a, "b": b}, (0.01,), 50, seed=7)
    second = v.subject_bootstrap({"a": a, "b": b}, (0.01,), 50, seed=7)
    assert np.array_equal(first["a"]["eer"], second["a"]["eer"]) and len(first["b"]["tar_at_0.01"]) == 50
    low, high = v.interval(first["a"]["eer"])
    assert low <= v.eer(v.rates(a, v.Grid(a))) <= high
