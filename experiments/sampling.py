import random
import typing

import nltk.tokenize

import data


def jaccard_distance(left: data.DocumentBase, right: data.DocumentBase) -> float:
    left_tokens = nltk.tokenize.word_tokenize(left.text)
    right_tokens = nltk.tokenize.word_tokenize(right.text)
    intersection = set(left_tokens).intersection(right_tokens)

    total_num_tokens = len(left_tokens) + len(right_tokens)
    num_same_tokens = len(intersection)

    return num_same_tokens / (total_num_tokens + num_same_tokens)


def random_sample_examples(
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


def similarity_sample_examples(
    documents: typing.List[data.DocumentBase],
    test_document: data.DocumentBase,
    num_examples: int,
) -> typing.List[data.DocumentBase]:
    if num_examples <= 0:
        return []
    documents_by_id: typing.Dict[str, data.DocumentBase] = {}
    for d in documents:
        if d.id != test_document.id:
            documents_by_id[d.id] = d
    assert test_document is not None
    document_similarities: typing.Dict[str, float] = {
        d.id: jaccard_distance(d, test_document) for d in documents_by_id.values()
    }

    candidates: typing.List[str, float] = sorted(
        document_similarities.items(), key=lambda x: x[1], reverse=True
    )

    return [documents_by_id[doc_id] for doc_id, _ in candidates[0:num_examples]]


def generate_folds(
    documents: typing.List[data.DocumentBase],
    num_examples: int,
    strategy: typing.Literal["random", "similarity"],
    seed: int = None,
) -> typing.List[typing.Dict[str, typing.List[str]]]:
    folds = []
    rng = random.Random(seed)
    for d in documents:
        if strategy == "random":
            examples = random_sample_examples(documents, d.id, num_examples, rng)
        elif strategy == "similarity":
            examples = similarity_sample_examples(documents, d, num_examples)
        else:
            raise ValueError(f'Unknown sampling strategy "{strategy}".')
        folds.append(
            {
                "train": [d.id for d in examples],
                "test": [d.id],
            }
        )

    return folds


def generate_sentence_constraint_folds(
    documents: typing.List[data.VanDerAaDocument],
    num_examples: int,
    strategy: typing.Literal["random", "similarity"],
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
                        documents, d.id, num_examples, strategy, rng
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
    strategy: typing.Literal["random", "similarity"],
    rng: random.Random,
):
    test_document = None
    for document in documents:
        if document.id == test_document_id:
            test_document = document
            break
    assert test_document is not None
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
        if strategy == "similarity":
            examples.extend(
                similarity_sample_examples(docs, test_document, num_examples)
            )
        else:
            examples.extend(rng.sample(docs, num_examples))
    return examples
