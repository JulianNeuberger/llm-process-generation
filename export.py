import json
import pathlib
import typing
import random

import data
import numpy as np

# set up indexes of SAP-SAM and PET
test_case: str = "test#2"

SAP_SAM_AMOUNT = 203 # first 207 lines of jsonl are SAP-SAM Models
PET_AMOUNT = 41 # the following 41 lines are PET Models
N_SAMPLES = 100 # amount of samples that will be taken from SAP-SAM Models for learning

assert 10 <= N_SAMPLES <= SAP_SAM_AMOUNT, f"Incorrect amount of samples. At least 10, at most {SAP_SAM_AMOUNT}"

def to_jerex(dataset: typing.List[data.PetDocument], out_dir: typing.Union[str, pathlib.Path]) -> None:
    def token_sentence_idx(token: data.PetToken, document: data.PetDocument) -> int:
        return document.sentences[token.sentence_index].index(token)

    def dump_doc(document: data.PetDocument) -> typing.Dict:
        vertex_set: list[list[dict[str, typing.Any]]] = list()
        mention_index_to_vertex_index: typing.Dict[int, int] = {}
        for entity in document.entities:
            vertex: list[dict[str, typing.Any]] = list()
            for mention_index in entity.mention_indices:
                mention_index_to_vertex_index[mention_index] = len(vertex_set)
                mention = document.mentions[mention_index]
                start_in_sentence = token_sentence_idx(document.tokens[mention.token_document_indices[0]], document)
                vertex.append({
                    "sent_id": document.tokens[mention.token_document_indices[0]].sentence_index,
                    "type": mention.type,
                    "pos": [
                        start_in_sentence,
                        start_in_sentence + len(mention.token_document_indices) + 1
                    ],
                    "name": mention.text(document)
                })
            vertex_set.append(vertex)

        labels: typing.Dict[typing.Tuple[int, int], typing.Dict] = {}
        for relation in document.relations:
            head = mention_index_to_vertex_index[relation.head_mention_index]
            tail = mention_index_to_vertex_index[relation.tail_mention_index]

            key = (head, tail)
            if key not in labels:
                labels[key] = {
                    "r": relation.type,
                    "h": head,
                    "t": tail,
                    "evidence": []
                }

            head_mention = document.mentions[relation.head_mention_index]
            tail_mention = document.mentions[relation.tail_mention_index]

            head_evidence = document.tokens[head_mention.token_document_indices[0]].sentence_index
            tail_evidence = document.tokens[tail_mention.token_document_indices[0]].sentence_index
            if head_evidence not in labels[key]["evidence"]:
                labels[key]["evidence"].append(head_evidence)
            if tail_evidence not in labels[key]["evidence"]:
                labels[key]["evidence"].append(tail_evidence)

        document_dict = {
            "vertexSet": vertex_set,
            "labels": list(labels.values()),
            "title": document.name,
            "sents": [[t.text for t in s] for s in document.sentences]
        }

        return document_dict

    lines = [dump_doc(d) for d in dataset]

    # define all indexes that we have for SAP-SAM and PET
    sap_sam_indexes = list(range(0, SAP_SAM_AMOUNT))
    pet_indexes = list(range(SAP_SAM_AMOUNT, len(lines)))
    # random.shuffle(lines)

    # test case #1
    # Train JEREX using SAP-SAM models only and check results using the whole PET-Dataset
    if test_case == "test#1":
        # define amounts of samples from SAP-SAM Dataset
        num_train = int(N_SAMPLES * 0.7) # int(len(lines) * 0.7)
        num_dev = int(N_SAMPLES * 0.3) # int(len(lines) * 0.1)
        # find indexes of train and validation dataset
        train_indexes = set(np.random.choice(sap_sam_indexes, size=num_train, replace=False))
        # exclude indexes that were selected for training
        resting_indexes = list(set(sap_sam_indexes).difference(train_indexes))
        dev_indexes = set(np.random.choice(resting_indexes, size=num_dev, replace=False))
        test_indexes = set(pet_indexes)
    
    # test case #2
    # Train JEREX using some small amount of PET models and additional amount of SAP-SAM models, test with the same set of PET-Models
    elif test_case == "test#2":
        # define a seed to always split PET-Dataset in the same manner
        random.seed(42)
        # split PET-Dataset
        random.shuffle(pet_indexes) # shuffle indexes
        num_train_pet = int(PET_AMOUNT * 0.5)
        num_dev_pet = int(PET_AMOUNT * 0.1)
        # get indexes of PET for future selection
        pet_train_indexes = set(pet_indexes[:num_train_pet])
        pet_dev_indexes = set(pet_indexes[num_train_pet: num_train_pet + num_dev_pet])
        pet_test_indexes = set(pet_indexes[num_train_pet + num_dev_pet:])

        # now select models from SAP-SAM
        np.random.seed(seed=None)
        num_train_sap_sam = int(N_SAMPLES * 0.7) # int(len(lines) * 0.7)
        num_dev_sap_sam = int(N_SAMPLES * 0.3) # int(len(lines) * 0.1)
        # select indexes for SAP-SAM Models
        sap_sam_train_indexes = set(np.random.choice(sap_sam_indexes, size=num_train_sap_sam, replace=False))
        # exclude indexes that were selected for training
        resting_indexes = list(set(sap_sam_indexes).difference(sap_sam_train_indexes))
        sap_sam_dev_indexes = set(np.random.choice(resting_indexes, size=num_dev_sap_sam, replace=False))

        # join all indexes
        train_indexes = sap_sam_train_indexes.union(pet_train_indexes)
        dev_indexes = sap_sam_dev_indexes.union(pet_dev_indexes)
        test_indexes = set(pet_test_indexes)

    # split all documents into train, validation and test sets
    train = [lines[idx] for idx in train_indexes]
    dev = [lines[idx] for idx in dev_indexes]
    test = [lines[idx] for idx in test_indexes]

    print("Train size:", len(train), "samples.")
    print("Validation size:", len(dev), "samples.")
    print("Test size:", len(test), "samples.")

    out_file = pathlib.Path(out_dir) / "train.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(train, f)

    out_file = pathlib.Path(out_dir) / "dev.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(dev, f)

    out_file = pathlib.Path(out_dir) / "test.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(test, f)

    entity_types = set(m.type for d in dataset for m in d.mentions)
    relation_types = set(r.type for d in dataset for r in d.relations)

    types = {
        "entities": {
            t: {"short": t, "verbose": t} for t in entity_types
        },
        "relations": {
            t: {"short": t, "verbose": t, "symmetric": False} for t in relation_types
        }
    }

    types_file = pathlib.Path(out_dir) / "types.json"
    with open(types_file, "w", encoding="utf8") as f:
        json.dump(types, f)


if __name__ == "__main__":
    in_path = pathlib.Path(__file__).parent / "res" / "data" / "annotate" / "sap sam" / "sap_sam_models_512_tokens.jsonl"
    dataset = data.PetImporter(str(in_path)).do_import()
    out_path = pathlib.Path(__file__).parent / "res" / "data" / "jerex" / test_case
    out_path.mkdir(exist_ok=True, parents=True)
    to_jerex(dataset, out_path)
