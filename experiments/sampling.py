import random
import typing

import data


def sample_examples(
    documents: typing.List[data.DocumentBase], test_document_id: str, num_examples: int
) -> typing.List[data.DocumentBase]:
    if num_examples == 0:
        return []
    documents_by_id = {d.id: d for d in documents if d.id != test_document_id}
    example_ids = random.sample(documents_by_id.keys(), num_examples)
    return [documents_by_id[i] for i in example_ids]


def generate_folds(
    documents: typing.List[data.DocumentBase], num_examples: int
) -> typing.List[typing.Dict[str, typing.List[str]]]:
    folds = []
    for d in documents:
        folds.append(
            {
                "train": [d.id for d in sample_examples(documents, d.id, num_examples)],
                "test": [d.id],
            }
        )
    return folds
