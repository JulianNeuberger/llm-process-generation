import json
import pathlib
import random
import typing

import data


def create_files(out_dir: typing.Union[str, pathlib.Path], train: typing.List[typing.Dict], dev: typing.List[typing.Dict], test: typing.List[typing.Dict]):
    out_file = pathlib.Path(out_dir) / "train.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(train, f)

    out_file = pathlib.Path(out_dir) / "dev.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(dev, f)

    out_file = pathlib.Path(out_dir) / "test.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(test, f)


def types_doc(dataset: typing.List[data.PetDocument], out_dir: typing.Union[str, pathlib.Path]):
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

def token_sentence_idx(token: data.PetToken, document: data.PetDocument) -> int:
    return document.sentences[token.sentence_index].index(token)


def dump_doc(document: data.PetDocument) -> typing.Dict:
    vertex_set = []
    mention_index_to_vertex_index: typing.Dict[int, int] = {}
    for entity in document.entities:
        if len(entity.mention_indices) == 0:
            print("0 mentions found: Ignoring")
            print(document.name)
            print(entity)
            continue
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

def to_jerex_plain(dataset: typing.List[data.PetDocument], out_dir: typing.Union[str, pathlib.Path]) -> None:
    out_dir = pathlib.Path(out_dir) / "pet_pet"
    out_dir.mkdir(exist_ok=True, parents=True)
    lines = [dump_doc(d) for d in dataset]
    random.shuffle(lines)
    test_list, pet_list = get_pet_split(lines)

    num_train = int(len(pet_list) * 0.9)

    train = pet_list[0: num_train]
    dev = pet_list[num_train:]
    test = test_list

    create_files(out_dir, train, dev, test)
    types_doc(dataset, out_dir)


def to_jerex_80(dataset: typing.List[data.PetDocument], pet_dataset: typing.List[data.PetDocument],out_dir: typing.Union[str, pathlib.Path]) -> None:
    out_dir = pathlib.Path(out_dir) / "pet_80"
    out_dir.mkdir(exist_ok=True, parents=True)
    lines = [dump_doc(d) for d in dataset]
    random.shuffle(lines)
    pet_lines = [dump_doc(d) for d in pet_dataset]
    random.shuffle(pet_lines)

    num_pet_train = int(len(pet_lines) * 0.8)
    pet_train = pet_lines[0:num_pet_train]
    pet_test = pet_lines[num_pet_train:]
    lines_comb = lines + pet_train

    num_train = int(len(lines_comb) * 0.9)
    num_dev = int(len(lines_comb) * 0.1)

    train = lines_comb[0: num_train]
    dev = lines_comb[num_train: num_train + num_dev]

    create_files(out_dir, train, dev, pet_test)
    types_doc(dataset, out_dir)

def to_jerex_rough(pet_dataset: typing.List[data.PetDocument], rough_dataset: typing.List[data.PetDocument], out_dir: typing.Union[str, pathlib.Path]) -> None:
    out_dir = pathlib.Path(out_dir) / "pet_rough"
    out_dir.mkdir(exist_ok=True, parents=True)
    pet_lines = [dump_doc(d) for d in pet_dataset]
    pet_test, pet_train = get_pet_split(pet_lines)
    lines = []
    if rough_dataset is not None:
        rough_lines = [dump_doc(d) for d in rough_dataset]
        lines = lines + rough_lines
    random.shuffle(lines)
    num_train_lines = int(len(lines) * 0.9)
    train = lines[0: num_train_lines]
    dev = lines[num_train_lines:]
    random.shuffle(train)
    random.shuffle(lines)
    print("Train len", len(train))
    print("Dev len", len(dev))
    create_files(out_dir, train, dev, pet_test)
    types_doc(pet_dataset, out_dir)
def to_jerex_pet_rough_fine_mul(pet_dataset: typing.List[data.PetDocument], rough_dataset: typing.List[data.PetDocument], fine_dataset, out_dir: typing.Union[str, pathlib.Path]) -> None:
    for i in range(1, 6):
        dir_name = "Pet_Fine_Rough" + str(i)
        out_dir_run = pathlib.Path(out_dir) / dir_name
        out_dir_run.mkdir(exist_ok=True, parents=True)
        pet_lines = [dump_doc(d) for d in pet_dataset]
        pet_test, pet_train = get_pet_split(pet_lines, 0)
        random.shuffle(pet_train)
        lines = []
        if fine_dataset is not None:
            fine_lines = [dump_doc(d) for d in fine_dataset]
            random.shuffle(fine_lines)
            lines = lines + fine_lines
        if rough_dataset is not None:
            rough_lines = [dump_doc(d) for d in rough_dataset]
            random.shuffle(rough_lines)
            lines = lines + rough_lines

        num_train_lines = int(len(lines) * 0.9)
        num_train_pet = int(len(pet_train) * 0.9)
        train = pet_train[0:num_train_pet] + lines[0: num_train_lines]
        dev = pet_train[num_train_pet:] + lines[num_train_lines:]
        print("Train len", len(train))
        print("Dev len", len(dev))
        create_files(out_dir_run, train, dev, pet_test)
        types_doc(pet_dataset, out_dir_run)
def to_jerex_pet_rough_fine(pet_dataset: typing.List[data.PetDocument], rough_dataset: typing.List[data.PetDocument], fine_dataset, out_dir: typing.Union[str, pathlib.Path]) -> None:

    out_dir_run = pathlib.Path(out_dir) / "new"
    #out_dir_run.mkdir(exist_ok=True, parents=True)
    pet_lines = [dump_doc(d) for d in pet_dataset]
    pet_test, pet_train = get_pet_split(pet_lines,5)
    random.shuffle(pet_train)

    lines = []
    if rough_dataset is not None:
        rough_lines = [dump_doc(d) for d in rough_dataset]
        random.shuffle(rough_lines)
        lines = lines + rough_lines
    if fine_dataset is not None:
        fine_lines = [dump_doc(d) for d in fine_dataset]
        random.shuffle(fine_lines)
        lines = lines + fine_lines

    num_train_lines = int(len(lines) * 0.9)
    num_train_pet = int(len(pet_train) * 0.9)
    train = pet_train[0:num_train_pet] + lines[0: num_train_lines]
    dev = pet_train[num_train_pet:] + lines[num_train_lines:]
    print("Train len", len(train))
    print("Dev len", len(dev))
    create_files(out_dir_run, train, dev, pet_test)
    types_doc(pet_dataset, out_dir_run)

def to_jerex_validating() -> None:
    out_dir = pathlib.Path(__file__).parent.parent / "res" / "data" / "jerex" / "validate"
    out_dir.mkdir(exist_ok=True, parents=True)
    load_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "annotate" / "pipe" / "run_10" / "2_annotated" / "Board_Meeting_Preparation_and_Execution.json"
    dataset = data.PetImporter(load_path).do_import()
    validate = [dump_doc(d) for d in dataset]

    out_file = pathlib.Path(out_dir) / "validate.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(validate, f)


def get_pet_split(pet_dataset: typing.List[data.PetDocument], number: int):
    counter = 1
    if number == 5:
        number = 0
    test_list = list()
    data_pet = list()
    for x in pet_dataset:
        if counter % 5 == number:
            test_list.append(x)
        else:
            data_pet.append(x)
        counter += 1
    return test_list, data_pet


if __name__ == "__main__":
    in_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "annotate" / "pipe" / "run_02_jerex" /"complete.jsonl"
    data_set = data.PetImporter(in_path).do_import()
    test_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "pet" / "all.new.jsonl"
    pet_data_set = data.PetImporter(test_path).do_import()
    out_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "jerex" / "test_randomized"
    rough_data_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "annotate" / "pipe" / "run_jerex_02" / "complete.jsonl"
    rough_data_set = data.PetImporter(rough_data_path).do_import()
    fine_path = pathlib.Path(__file__).parent.parent / "res" / "data" / "annotate" / "pipe" / "run_10" / "2_annotated" / "hand_made.jsonl"
    fine_data_set = data.PetImporter(fine_path).do_import()
    out_path.mkdir(exist_ok=True, parents=True)
    to_jerex_pet_rough_fine_mul(pet_dataset=pet_data_set, rough_dataset=rough_data_set, fine_dataset=fine_data_set, out_dir=out_path)
    #to_jerex_rough(pet_data_set,rough_data_set, out_path)
    #to_jerex_validating()
    #test, pet = get_pet_split(pet_dataset)
    #print("Pet", len(pet), "test", len(test))
