from typing import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr


def fragment_scores(node_mask: np.ndarray):
    """Reduce a (n_fragments, n_features) mask to a scalar per fragment.

    Sums across the feature axis. Sum aggregation preserves the sign of
    contributions, which is important when attributions can be negative
    (e.g. Integrated Gradients).

    Args:
        node_mask: 2-D numpy array of shape (n_fragments, n_features).

    Returns:
        1-D numpy array of length n_fragments.
    """
    arr = np.asarray(node_mask, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr.sum(axis=1)


def gini(values: np.ndarray):
    """Gini coefficient on absolute importance values.

    A value of 0 means perfectly uniform attribution; 1 means all the
    importance is concentrated on a single fragment. The metric uses
    absolute values so it is well-defined when attributions can be
    negative.

    Args:
        values: 1-D numpy array of fragment importance scores.

    Returns:
        Float in [0, 1]. Returns 0.0 for arrays with fewer than
        two entries or with zero total absolute mass.
    """
    arr = np.abs(np.asarray(values, dtype=float)).ravel()
    n = arr.size
    if n < 2:
        return 0.0

    total = arr.sum()
    if total <= 0:
        return 0.0

    sorted_arr = np.sort(arr)
    ranks = np.arange(1, n + 1)
    numerator = ((2 * ranks - n - 1) * sorted_arr).sum()
    return float(numerator / (n * total))


def mean_gini(per_mol_scores: Sequence[np.ndarray]):
    """Mean Gini coefficient across a population of molecules.

    Args:
        per_mol_scores: Sequence of 1-D numpy arrays, one per molecule.

    Returns:
        Float; 0.0 if the input sequence is empty.
    """
    if len(per_mol_scores) == 0:
        return 0.0
    return float(np.mean([gini(s) for s in per_mol_scores]))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                  seed: int = 13):
    """95% percentile bootstrap CI of the mean of a 0/1 array."""
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return (0.0, 0.0)
    boot = rng.choice(values, size=(n_boot, n), replace=True)
    means = boot.mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return (lo, hi)


def fragment_hit_rate(top_fragments: Sequence[str],
                      classifier: Callable[[str], str],
                      class_names: Sequence[str],
                      n_boot: int = 1000):
    """Fraction of molecules whose top fragment matches a known class.

    Args:
        top_fragments: One fragment SMILES per molecule (the argmax of
            its fragment-importance vector).
        classifier: Callable that maps a fragment SMILES to a class
            name string or None.
        class_names: All valid class names. Used to populate a
            per-class breakdown even when some classes never matched.
        n_boot: Bootstrap resamples for the overall-rate CI.

    Returns:
        Dict with keys overall, ci_low, ci_high,
        per_class (mapping class name -> fraction of molecules),
        and n (population size).
    """
    n = len(top_fragments)
    if n == 0:
        return {"overall": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "per_class": {c: 0.0 for c in class_names},
                "n": 0}

    matches = np.zeros(n, dtype=int)
    per_class = {c: 0 for c in class_names}
    for i, frag in enumerate(top_fragments):
        label = classifier(frag)
        if label is None:
            continue
        matches[i] = 1
        if label in per_class:
            per_class[label] += 1

    overall = float(matches.mean())
    lo, hi = _bootstrap_ci(matches.astype(float), n_boot=n_boot)
    per_class_rate = {c: per_class[c] / n for c in class_names}

    return {"overall": overall,
            "ci_low": lo,
            "ci_high": hi,
            "per_class": per_class_rate,
            "n": n}


def spearman_cross_explainer(scores_a: Sequence[np.ndarray],
                             scores_b: Sequence[np.ndarray]):
    """Mean per-molecule Spearman correlation between two explainers.

    Molecules with fewer than two fragments contribute nothing (Spearman
    is undefined). Molecules where one of the score vectors is constant
    likewise return NaN from scipy and are skipped.

    Args:
        scores_a: One importance vector per molecule from explainer A.
        scores_b: Same population, same order, from explainer B.

    Returns:
        Dict with mean (float), std (float), and n_used
        (count of molecules contributing to the mean).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    rhos = []
    for a, b in zip(scores_a, scores_b):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.size < 2 or a.size != b.size:
            continue
        rho, _ = spearmanr(a, b)
        if rho is None or np.isnan(rho):
            continue
        rhos.append(float(rho))

    if not rhos:
        return {"mean": 0.0, "std": 0.0, "n_used": 0}
    return {"mean": float(np.mean(rhos)),
            "std": float(np.std(rhos)),
            "n_used": len(rhos)}


def top_fragment(score_vec: np.ndarray, fragments: Sequence[str]):
    """Return the fragment SMILES with the largest absolute score.

    Args:
        score_vec: 1-D fragment-importance vector.
        fragments: Aligned list of fragment SMILES.

    Returns:
        SMILES of the argmax-|score| fragment, or None if empty.
    """
    arr = np.asarray(score_vec, dtype=float).ravel()
    if arr.size == 0 or len(fragments) == 0:
        return None
    idx = int(np.argmax(np.abs(arr)))
    return fragments[idx]
