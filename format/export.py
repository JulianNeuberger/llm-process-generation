import json
import pathlib
import random
import typing

import data


def to_jerex(dataset: typing.List[data.PetDocument], out_dir: typing.Union[str, pathlib.Path]) -> None:
    def token_sentence_idx(token: data.PetToken, document: data.PetDocument) -> int:
        return document.sentences[token.sentence_index].index(token)

    def dump_doc(document: data.PetDocument) -> typing.Dict:
        vertex_set = []
        mention_index_to_vertex_index: typing.Dict[int, int] = {}
        for entity in document.entities:
            vertex = []
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
    random.shuffle(lines)

    num_train = int(len(lines) * 0.7)
    num_dev = int(len(lines) * 0.1)

    train = lines[0: num_train]
    dev = lines[num_train: num_train + num_dev]
    test = lines[num_train + num_dev: ]

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
    in_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "annotate" / "tests" / "Inventory_Replenishment_Process.json"
    dataset = data.PetImporter(in_path).do_import()
    out_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "jerex"
    out_path.mkdir(exist_ok=True, parents=True)
    to_jerex(dataset, out_path)
