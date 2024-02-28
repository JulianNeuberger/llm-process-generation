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
            print(f"--- name: {p.name}, id: {p.id} ------------")
        case_stats = constraint_slot_filling_stats(
            t, true=t.constraints, pred=p.constraints, verbose=verbose
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
    document: data.VanDerAaDocument,
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

    non_ok = [p for p in pred if p not in best_matches.keys()]
    ok = list(best_matches.values())
    missing = [t for t in true if t not in best_matches.values()]

    if verbose:
        print_sets(document.id, document.text, true, pred, ok, non_ok, missing)
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


def pretty_print_tuple(
    document: data.DocumentBase, element_as_tuple: typing.Tuple, attribute: str
):
    if attribute == "mentions":
        mention = data.PetMention(
            type=element_as_tuple[0], token_document_indices=list(element_as_tuple[1:])
        )
        return f"({mention.type}, {mention.text(document)}, {mention.token_document_indices})"
    if attribute == "entities":
        entity = data.PetEntity(mention_indices=list(element_as_tuple))
        return [
            f"({document.mentions[i].type}, {document.mentions[i].text(document)}, {document.mentions[i].token_document_indices})"
            for i in entity.mention_indices
        ]
    if attribute == "relations":
        relation = data.PetRelation(
            type=element_as_tuple[0],
            head_mention_index=element_as_tuple[1],
            tail_mention_index=element_as_tuple[2],
        )
        head = document.mentions[relation.head_mention_index]
        tail = document.mentions[relation.tail_mention_index]

        return f"{head.text(document)} -{relation.type}> {tail.text(document)}"
    raise AssertionError()


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

        true = set([e.to_tuple() for e in true_attribute])
        pred = set([e.to_tuple() for e in pred_attribute])
        ok = true.intersection(pred)
        non_ok = pred.difference(true)
        missing = true.difference(pred)

        if len(true) != len(true_attribute):
            # contains identical values, need to use lists
            true = [e.to_tuple() for e in true_attribute]
            pred = [e.to_tuple() for e in pred_attribute]
            true_candidates = [t for t in true]
            ok = []
            non_ok = []
            for cur in pred:
                if cur in true_candidates:
                    true_candidates.remove(cur)
                    ok.append(cur)
                    continue
                non_ok.append(cur)
            missing = true_candidates

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, t),
            true,
            "gold",
        )
        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, p),
            pred,
            "pred",
        )

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _get_ner_tag_for_tuple(attribute, e, p),
            ok,
            "ok",
        )

        if verbose:
            print_sets(
                p.id,
                p.text,
                [pretty_print_tuple(p, t, attribute) for t in true],
                [pretty_print_tuple(p, t, attribute) for t in pred],
                [pretty_print_tuple(p, t, attribute) for t in ok],
                [pretty_print_tuple(p, t, attribute) for t in non_ok],
                [pretty_print_tuple(p, t, attribute) for t in missing],
            )

    return {
        tag: Stats(num_pred=p, num_gold=g, num_ok=o)
        for tag, (g, p, o) in stats_by_tag.items()
    }


def print_sets(
    document_id: str,
    document_text: str,
    true: typing.List[str],
    pred: typing.List[str],
    ok: typing.List[str],
    non_ok: typing.List[str],
    missing: typing.List[str],
):
    print(f"=== {document_id} " + "=" * 150)
    print(document_text)
    print("-" * 100)
    print(f"{len(true)} x true")
    print("\n".join(true))
    print("-" * 100)
    print()
    print(f"{len(pred)} x pred")
    print("\n".join(pred))
    print("-" * 100)
    print()
    print(f"{len(ok)} x ok")
    print("\n".join(ok))
    print("-" * 100)
    print()
    print(f"{len(non_ok)} x non ok")
    print("\n".join(non_ok))
    print("-" * 100)
    print()
    print(f"{len(missing)} x missing")
    print("\n".join(missing))
    print()
    print("=" * 150)
    print()
