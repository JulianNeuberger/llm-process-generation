import typing

from eval.metrics import (
    relation_f1_stats,
    mentions_f1_stats,
    entity_f1_stats,
    constraint_f1_stats,
    Scores,
    Stats,
)


def stats_to_scores(stats: typing.Dict[str, Stats]) -> typing.Dict[str, Scores]:
    return {tag: Scores.from_stats(stats) for tag, stats in stats.items()}


def average_scores(
    stats: typing.Dict[str, Stats], strategy: typing.Literal["micro", "macro"]
) -> Scores:
    if strategy == "micro":
        combined_stats = sum(stats.values(), metrics.Stats(0, 0, 0))
        return metrics.Scores.from_stats(combined_stats)
    if strategy == "macro":
        scores = stats_to_scores(stats)
        return sum(scores.values(), metrics.Scores(0, 0, 0)) / len(scores)
    raise ValueError(f"Unknown averaging mode {strategy}.")
