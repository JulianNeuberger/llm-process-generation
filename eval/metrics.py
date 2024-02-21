import dataclasses
import typing

import data


@dataclasses.dataclass
class Scores:
    p: float
    r: float
    f1: float

    @staticmethod
    def from_stats(stats: "Stats") -> "Scores":
        return Scores(p=stats.precision, r=stats.recall, f1=stats.f1)

    def __add__(self, other):
        if type(other) != Scores:
            raise TypeError(f"Can not add Scores and {type(other)}")
        return Scores(p=self.p + other.p, r=self.r + other.r, f1=self.f1 + other.f1)

    def __truediv__(self, other):
        return Scores(p=self.p / other, r=self.r / other, f1=self.f1 / other)


@dataclasses.dataclass
class Stats:
    num_pred: float
    num_gold: float
    num_ok: float

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if precision + recall == 0.0:
            return 0
        return 2 * precision * recall / (precision + recall)

    @property
    def precision(self) -> float:
        if self.num_pred == 0 and self.num_gold == 0:
            return 1.0
        elif self.num_pred == 0 and self.num_gold != 0:
            return 0.0
        else:
            return self.num_ok / self.num_pred

    @property
    def recall(self) -> float:
        if self.num_gold == 0 and self.num_pred == 0:
            return 1.0
        elif self.num_gold == 0 and self.num_pred != 0:
            return 0.0
        else:
            return self.num_ok / self.num_gold

    def __add__(self, other):
        if type(other) != Stats:
            raise TypeError(f"Can not add Stats and {type(other)}")
        return Stats(
            num_pred=self.num_pred + other.num_pred,
            num_gold=self.num_gold + other.num_gold,
            num_ok=self.num_ok + other.num_ok,
        )


def constraint_f1_stats(
    *,
    predicted_documents: typing.List[data.VanDerAaDocument],
    ground_truth_documents: typing.List[data.VanDerAaDocument],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    ret: typing.Dict[str, Stats] = {}
    assert len(predicted_documents) == len(ground_truth_documents)
    for p, t in zip(predicted_documents, ground_truth_documents):
        if verbose:
            print(f"--- {p.name} ------------")
        case_stats = constraint_slot_filling_stats(
            true=t.constraints, pred=p.constraints, verbose=verbose
        )
        if verbose:
            print()
            print()
        for tag, stats in case_stats.items():
            if tag not in ret:
                ret[tag] = Stats(0, 0, 0)
            ret[tag] += stats
    return ret


def relation_f1_stats(
    *,
    predicted_documents: typing.List[data.PetDocument],
    ground_truth_documents: typing.List[data.PetDocument],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="relations",
        verbose=verbose,
    )


def mentions_f1_stats(
    *,
    predicted_documents: typing.List[data.PetDocument],
    ground_truth_documents: typing.List[data.PetDocument],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="mentions",
        verbose=verbose,
    )


def entity_f1_stats(
    *,
    predicted_documents: typing.List[data.PetDocument],
    ground_truth_documents: typing.List[data.PetDocument],
    only_tags: typing.List[str],
    min_num_mentions: int = 1,
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    predicted_documents = [d.copy() for d in predicted_documents]
    for d in predicted_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions and e.get_tag(d) in only_tags
        ]

    ground_truth_documents = [d.copy() for d in ground_truth_documents]
    for d in ground_truth_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions and e.get_tag(d) in only_tags
        ]

    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="entities",
        verbose=verbose,
    )


def _add_to_stats_by_tag(
    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]],
    get_tag: typing.Callable[[typing.Any], str],
    object_list: typing.Iterable,
    stat: str,
):
    assert stat in ["gold", "pred", "ok"]
    for e in object_list:
        tag = get_tag(e)
        if tag not in stats_by_tag:
            stats_by_tag[tag] = (0, 0, 0)
        prev_stats = stats_by_tag[tag]
        if stat == "gold":
            stats_by_tag[tag] = (prev_stats[0] + 1, prev_stats[1], prev_stats[2])
        elif stat == "pred":
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1] + 1, prev_stats[2])
        else:
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1], prev_stats[2] + 1)
    return stats_by_tag


def _get_ner_tag_for_tuple(
    element_type: str, element: typing.Tuple, document: data.DocumentBase
) -> str:
    assert element_type in ["mentions", "relations", "entities", "constraints"]
    assert type(element) == tuple
    if element_type == "entities":
        assert (
            type(document) == data.PetDocument
        ), "Mentions currently only supported for PET documents."
        mentions = [document.mentions[i] for i in element]
        return mentions[0].type
    return element[0]


def constraint_slot_filling_stats(
    *,
    true: typing.List[data.VanDerAaConstraint],
    pred: typing.List[data.VanDerAaConstraint],
    verbose: bool,
) -> typing.Dict[str, Stats]:
    best_matches: typing.Dict[data.VanDerAaConstraint, data.VanDerAaConstraint] = {}

    for score in range(4, -1, -1):
        for p in pred:
            for t in true:
                num_correct_slots = p.correct_slots(t)
                if num_correct_slots != score:
                    continue
                if p in best_matches.keys():
                    continue
                if t in best_matches.values():
                    continue
                best_matches[p] = t
                break
    has_unmatched_predictions = False
    for p in pred:
        if p not in best_matches and verbose:
            has_unmatched_predictions = True
            print(f"Found no match for constraint {p.to_tuple()}")
    if verbose:
        if not has_unmatched_predictions:
            print("Matched all predictions.")
        else:
            candidates = [t for t in true if t not in best_matches.values()]
            if len(candidates) == 0:
                print(
                    f"No more ground truth candidates of the {len(true)} ones "
                    f"remained after matching the other predictions."
                )
            else:
                print("Remaining ground truth candidates are:")
                print([t.to_tuple() for t in candidates])
    stats_by_tag = {}

    for t in true:
        if t.type not in stats_by_tag:
            stats_by_tag[t.type] = Stats(0, 0, 0)
        stats_by_tag[t.type].num_gold += t.num_slots

    for p in pred:
        if p.type not in stats_by_tag:
            stats_by_tag[p.type] = Stats(0, 0, 0)
        stats_by_tag[p.type].num_pred += p.num_slots

    for p in best_matches:
        t = best_matches[p]
        if p.type not in stats_by_tag:
            stats_by_tag[p.type] = Stats(0, 0, 0)
        stats_by_tag[p.type].num_ok += p.correct_slots(t)

    if verbose:
        print()
        print(
            f"Ground truth has {sum([s.num_gold for s in stats_by_tag.values()])} slots."
        )
        print(
            f"Approach predicted {sum([s.num_pred for s in stats_by_tag.values()])} slots."
        )
        print(
            f"Of those {sum([s.num_ok for s in stats_by_tag.values()])} slots were correct."
        )

    return stats_by_tag


def _f1_stats(
    *,
    predicted_documents: typing.List[data.DocumentBase],
    ground_truth_documents: typing.List[data.DocumentBase],
    attribute: str,
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    assert attribute in ["mentions", "relations", "entities", "constraints"]
    assert len(predicted_documents) == len(ground_truth_documents)

    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]] = {}

    for p, t in zip(predicted_documents, ground_truth_documents):
        true_attribute = getattr(t, attribute)
        pred_attribute = getattr(p, attribute)

        true_as_set = set([e.to_tuple() for e in true_attribute])
        assert len(true_as_set) == len(
            true_attribute
        ), f"{len(true_as_set)}, {len(true_attribute)}, {true_as_set}, {true_attribute}"

        pred_as_set = set([e.to_tuple() for e in pred_attribute])

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, t),
            true_as_set,
            "gold",
        )
        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, p),
            pred_as_set,
            "pred",
        )

        ok_preds = true_as_set.intersection(pred_as_set)

        ok = [e.to_tuple() for e in pred_attribute if e.to_tuple() in ok_preds]

        non_ok = [
            e.to_tuple()
            for e in pred_attribute
            if e.to_tuple() not in true_as_set
            # if _get_ner_tag_for_tuple(attribute, e.to_tuple(), p).lower() == 'actor'
        ]

        if verbose:  # and len(non_ok) > 0:
            print(f"=== {t.id} " + "=" * 150)
            print(p.text)
            print("-" * 100)
            print("true")
            print([e for e in true_as_set])
            print("-" * 100)
            print()
            print("pred")
            print([e for e in pred_as_set])
            print("-" * 100)
            print()
            print("ok")
            print(ok)
            print("-" * 100)
            print()
            print("non ok")
            print(non_ok)
            print()
            print("=" * 150)
            print()

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, p),
            ok_preds,
            "ok",
        )

    return {
        tag: Stats(num_pred=p, num_gold=g, num_ok=o)
        for tag, (g, p, o) in stats_by_tag.items()
    }
