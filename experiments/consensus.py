import typing

import dataclasses
import json
import typing
from eval.metrics import Stats, _add_to_stats_by_tag, _tag, print_sets
import data
import eval
import experiments
import format
TDocument = typing.TypeVar("TDocument", bound=data.DocumentBase)
def consensus_2(
    current_result,
    compare_result,
    verbose: bool,
    print_only_tags: typing.Optional[typing.List[str]],
    overall_steps):
    current: typing.List[TDocument] = {}
    compare: typing.List[TDocument] = {}
    current = current_result
    compare = compare_result

    stats = {}
    if "mentions" in overall_steps:
        stats_by_tag, missing = mentions_f1_stats_list(
            current_documents=current,
            compare_documents=compare,
            verbose=verbose,
            print_only_tags=print_only_tags,
        )
        stats["mentions"] = stats_by_tag
    return stats, missing


def mentions_f1_stats_list(
    *,
    current_documents: typing.List[data.PetDocument],
    compare_documents: typing.List[data.PetDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Tuple[typing.Dict[str, Stats], typing.List]:
    assert len(current_documents) == len(compare_documents)

    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]] = {}

    for p, t in zip(current_documents, compare_documents):
        cur_attribute = getattr(p, "mentions")
        compare_attribute = getattr(t, "mentions")

        true = list(compare_attribute)
        curr = list(cur_attribute)
        compare_candidates = list(compare_attribute)
        ok = []
        non_ok = []

        for cur in curr:
            match: typing.Optional[data.DocumentBase] = None
            if isinstance(cur, data.HasCustomMatch):
                for candidate in compare_candidates:
                    if cur.match(candidate):
                        match = candidate
                        break
            else:
                try:
                    match_index = compare_candidates.index(cur)
                    match = compare_candidates[match_index]
                except ValueError:
                    pass

            if match is not None:
                compare_candidates.remove(match)
                ok.append(cur)
                continue
            non_ok.append(cur)
        missing = compare_candidates

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            true,
            "gold",
        )
        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            curr,
            "pred",
        )

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            ok,
            "ok",
        )

        if verbose and (len(non_ok) > 0 or len(missing) > 0):
            print_sets(
                t,
                {
                    "true": true,
                    "pred": curr,
                    # "ok": ok,
                    "non-ok": non_ok,
                    "missing": missing,
                },
                lambda e: _tag(t, e),
                print_only_tags,
            )

    return {
        tag: Stats(num_pred=p, num_gold=g, num_ok=o)
        for tag, (g, p, o) in stats_by_tag.items()
    }, missing




def main():
    cur_file = "res/answers/gpt-4o-mini/annotate/2024-10-03_19-08-36.json"

    return 0

