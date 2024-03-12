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
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="constraints",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def slot_filling(
    *,
    predicted_documents: typing.List[data.VanDerAaDocument],
    ground_truth_documents: typing.List[data.VanDerAaDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
):
    ret: typing.Dict[str, Stats] = {}
    assert len(predicted_documents) == len(ground_truth_documents)
    for p, t in zip(predicted_documents, ground_truth_documents):
        if verbose:
            print(f"--- name: {p.name}, id: {p.id} ------------")
        case_stats = constraint_slot_filling_stats(
            t,
            true=t.constraints,
            pred=p.constraints,
            print_only_tags=print_only_tags,
            verbose=verbose,
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
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="relations",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def mentions_f1_stats(
    *,
    predicted_documents: typing.List[data.PetDocument],
    ground_truth_documents: typing.List[data.PetDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="mentions",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def entity_f1_stats(
    *,
    predicted_documents: typing.List[data.PetDocument],
    ground_truth_documents: typing.List[data.PetDocument],
    calculate_only_tags: typing.List[str],
    min_num_mentions: int = 1,
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    calculate_only_tags = [t.lower() for t in calculate_only_tags]
    for d in predicted_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions
            and e.get_tag(d).lower() in calculate_only_tags
        ]

    ground_truth_documents = [d.copy([]) for d in ground_truth_documents]
    for d in ground_truth_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions
            and e.get_tag(d) in calculate_only_tags
        ]

    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="entities",
        print_only_tags=print_only_tags,
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


def constraint_slot_filling_stats(
    document: data.VanDerAaDocument,
    *,
    true: typing.List[data.VanDerAaConstraint],
    pred: typing.List[data.VanDerAaConstraint],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool,
) -> typing.Dict[str, Stats]:
    best_matches: typing.Dict[data.VanDerAaConstraint, data.VanDerAaConstraint] = {}

    for score in range(4, -1, -1):
        for p in pred:
            for t in true:
                if p.sentence_id != t.sentence_id:
                    continue

                num_correct_slots = p.correct_slots(t)
                if num_correct_slots != score:
                    continue
                if p in best_matches.keys():
                    continue
                if t in best_matches.values():
                    continue
                best_matches[p] = t
                break

    non_ok = [p for p in pred if p not in best_matches.keys()]
    missing = [t for t in true if t not in best_matches.values()]

    by_correct_slots = {}
    for p, t in best_matches.items():
        num_correct_slots = f"{p.correct_slots(t)} correct slots"
        if num_correct_slots not in by_correct_slots:
            by_correct_slots[num_correct_slots] = []
        by_correct_slots[num_correct_slots].append(p)

    if verbose:
        print_sets(
            document,
            {
                "true": true,
                "pred": pred,
                **by_correct_slots,
                "non-ok": non_ok,
                "missing": missing,
            },
            lambda c: c.type,
            print_only_tags,
        )
    stats_by_tag = {}

    for t in true:
        if t.type not in stats_by_tag:
            stats_by_tag[t.type] = Stats(0, 0, 0)
        stats_by_tag[t.type].num_gold += t.num_slots

    for p in pred:
        if p in best_matches:
            continue
        if p.type not in stats_by_tag:
            stats_by_tag[p.type] = Stats(0, 0, 0)
        stats_by_tag[p.type].num_pred += p.num_slots

    for p in best_matches:
        t = best_matches[p]
        if t.type not in stats_by_tag:
            stats_by_tag[t.type] = Stats(0, 0, 0)
        num_correct_slots = p.correct_slots(t)
        assert num_correct_slots <= t.num_slots
        stats_by_tag[t.type].num_ok += num_correct_slots
        stats_by_tag[t.type].num_pred += p.num_slots
    for _, s in stats_by_tag.items():
        assert s.num_ok <= s.num_gold

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


def _tag(
    document: data.DocumentBase,
    e: typing.Union[
        data.PetMention,
        data.PetEntity,
        data.PetRelation,
        data.VanDerAaConstraint,
        data.QuishpiMention,
    ],
) -> str:
    if isinstance(e, data.HasType):
        return e.type
    if type(e) == data.PetEntity:
        assert type(document) == data.PetDocument
        return e.get_tag(document)
    if type(e) == data.VanDerAaMention:
        return "action"
    raise AssertionError(f"Unknown type {type(e)}")


def _f1_stats(
    *,
    predicted_documents: typing.List[data.DocumentBase],
    ground_truth_documents: typing.List[data.DocumentBase],
    attribute: str,
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    assert attribute in ["mentions", "relations", "entities", "constraints"]
    assert len(predicted_documents) == len(ground_truth_documents)

    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]] = {}

    for p, t in zip(predicted_documents, ground_truth_documents):
        true_attribute = getattr(t, attribute)
        pred_attribute = getattr(p, attribute)

        true = list(true_attribute)
        pred = list(pred_attribute)
        true_candidates = list(true_attribute)
        ok = []
        non_ok = []
        for cur in pred:
            match: typing.Optional[data.DocumentBase] = None
            if isinstance(cur, data.HasCustomMatch):
                for candidate in true_candidates:
                    if cur.match(candidate):
                        match = candidate
                        break
            else:
                try:
                    match_index = true_candidates.index(cur)
                    match = true_candidates[match_index]
                except ValueError:
                    pass

            if match is not None:
                true_candidates.remove(match)
                ok.append(cur)
                continue
            non_ok.append(cur)
        missing = true_candidates

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            true,
            "gold",
        )
        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            pred,
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
                    "pred": pred,
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
    }


def print_sets(
    document: data.DocumentBase,
    sets: typing.Dict[str, typing.List[data.SupportsPrettyDump]],
    get_tag: typing.Callable[[typing.Any], str],
    print_only_tags: typing.Optional[typing.List[str]],
):
    print(f"=== {document.id} " + "=" * 150)
    print(document.text)
    print("-" * 100)

    for set_name, values in sets.items():
        values = [
            e
            for e in values
            if print_only_tags is None or get_tag(e) in print_only_tags
        ]
        print(f"{len(values)} x {set_name}")
        print("\n".join([e.pretty_dump(document) for e in values]))
        print("-" * 100)
        print()

    print("=" * 150)
    print()
