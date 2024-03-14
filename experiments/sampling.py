import random
import typing

import data


def sample_examples(
    documents: typing.List[data.DocumentBase],
    test_document_id: str,
    num_examples: int,
    rng: random.Random,
) -> typing.List[data.DocumentBase]:
    if num_examples <= 0:
        return []

    documents_by_id = {d.id: d for d in documents if d.id != test_document_id}
    example_ids = rng.sample(documents_by_id.keys(), num_examples)
    return [documents_by_id[i] for i in example_ids]


def generate_folds(
    documents: typing.List[data.DocumentBase],
    num_examples: int,
    seed: int = None,
) -> typing.List[typing.Dict[str, typing.List[str]]]:
    folds = []
    rng = random.Random(seed)
    for d in documents:
        folds.append(
            {
                "train": [
                    d.id for d in sample_examples(documents, d.id, num_examples, rng)
                ],
                "test": [d.id],
            }
        )

    return folds


def generate_sentence_constraint_folds(
    documents: typing.List[data.VanDerAaDocument],
    num_examples: int,
    seed: int = None,
) -> typing.List[typing.Dict[str, typing.List[str]]]:
    folds = []
    rng = random.Random(seed)
    for d in documents:
        folds.append(
            {
                "train": [
                    d.id
                    for d in sample_sentence_constraints_stratified(
                        documents, d.id, num_examples, rng
                    )
                ],
                "test": [d.id],
            }
        )

    return folds


def sample_sentence_constraints_stratified(
    documents: typing.List[data.VanDerAaDocument],
    test_document_id: str,
    num_examples: int,
    rng: random.Random,
):
    by_type: typing.Dict[str, typing.List[data.VanDerAaDocument]] = {}
    for d in documents:
        if d.id == test_document_id:
            continue
        for c in d.constraints:
            if c.type not in by_type:
                by_type[c.type] = []
            by_type[c.type].append(d)
    examples = []
    items = list(by_type.items())
    for c_type, docs in items:
        examples.extend(rng.sample(docs, num_examples))
    return examples
