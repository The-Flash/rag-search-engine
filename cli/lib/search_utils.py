BM25_K1 = 1.5
BM25_B = 0.75


def min_max_normalize(score: float, min_score: float, max_score: float) -> float:
    if min_score == max_score:
        return 1.0
    # (score - min_score) / (max_score - min_score)
    new_score = (score - min_score) / (max_score - min_score)
    return new_score
