import csv
import dataclasses
import os.path
import typing

import nltk

from data import base
from data.base import TDocument


def excel_col_to_index(col: str):
    assert len(col) == 1
    col = col.lower()
    return ord(col) - 97


ID_COL = excel_col_to_index("E")
NAME_COL = excel_col_to_index("B")
TEXT_COL = excel_col_to_index("F")
NUM_CONSTRAINTS_COL = excel_col_to_index("G")
NEGATIVE_COL = excel_col_to_index("H")
CONSTRAINT_1_COL = excel_col_to_index("I")
CONSTRAINT_2_COL = excel_col_to_index("L")
CONSTRAINT_3_COL = excel_col_to_index("O")


@dataclasses.dataclass
class VanDerAaDocument(base.DocumentBase, base.HasMentions["VanDerAaMention"]):
    name: str
    sentences: typing.List[str]
    constraints: typing.List["VanDerAaConstraint"]

    def __add__(self, other: "VanDerAaDocument"):
        assert self.id == other.id
        assert len(self.sentences) == len(other.sentences)
        assert self.sentences == other.sentences

        new_constraints = [c for c in other.constraints if c not in self.constraints]
        new_mentions = [m for m in other.mentions if m not in self.mentions]

        return VanDerAaDocument(
            id=self.id,
            text=self.text,
            name=self.name,
            sentences=self.sentences,
            constraints=self.constraints + new_constraints,
            mentions=self.mentions + new_mentions,
        )

    def copy(self, clear: typing.List[str]):
        constraints = []
        if "constraints" not in clear:
            constraints = [c.copy() for c in self.constraints]
        return VanDerAaDocument(
            id=self.id,
            text=self.text,
            name=self.name,
            constraints=constraints,
            sentences=self.sentences,
        )


@dataclasses.dataclass(eq=True, frozen=True)
class VanDerAaMention(base.SupportsPrettyDump["VanDerAaDocument"]):
    text: str

    def pretty_dump(self, document: TDocument) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class VanDerAaConstraint(base.SupportsPrettyDump["VanDerAaDocument"]):
    type: str
    head: str
    tail: typing.Optional[str]
    negative: bool
    sentence_id: int

    def pretty_dump(self, document: VanDerAaDocument) -> str:
        pretty = f'{"TRUE" if self.negative else "FALSE"}\t{self.type}\t{self.head}'
        if self.tail:
            pretty = f"{pretty}\t{self.tail}"
        return f"s: {self.sentence_id}\t{pretty}"

    def copy(self):
        return VanDerAaConstraint(
            type=self.type,
            negative=self.negative,
            head=self.head,
            tail=self.tail,
            sentence_id=self.sentence_id,
        )

    @property
    def num_slots(self):
        slots = 2
        if self.tail is not None:
            slots += 1
        if self.negative:
            slots += 1
        return slots

    @staticmethod
    def action_match(pred: typing.Optional[str], true: typing.Optional[str]) -> bool:
        if pred == true:
            return True
        if pred is None:
            return False
        if true is None:
            return False

        pred = pred.lower()
        true = true.lower()

        true_verb = true.split(" ")[0]
        if true_verb.lower() in pred:
            return True
        pred_tokens = nltk.word_tokenize(pred)
        pred_pos_tags: typing.Iterable[typing.Tuple[str, str]] = nltk.pos_tag(
            pred_tokens
        )
        pred_verb: typing.Optional[str] = None
        for token, pos in pred_pos_tags:
            if pos.startswith("VB"):
                pred_verb = token
                break

        if pred_verb is None:
            pred_verb = pred_tokens[0]
            # print(
            #     f"Expected action {pred} to have a verb in it, but nltk found none in "
            #     f"{pred_pos_tags}, guessing first one, which is '{pred_verb}'."
            # )

        if pred_verb.lower() in true:
            return True
        return False

    def correct_slots(self, true: "VanDerAaConstraint") -> int:
        res = 0
        if self.type.lower() == true.type.lower():
            res += 1
        if VanDerAaConstraint.action_match(self.head, true.head):
            res += 1
        if true.tail is not None:
            if VanDerAaConstraint.action_match(self.tail, true.tail):
                res += 1
        if self.negative and true.negative:
            res += 1
            if self.type.lower() != true.type.lower():
                res += 1
        return res


class VanDerAaImporter(base.BaseImporter[VanDerAaDocument]):
    def __init__(self, path_to_collection: str):
        self._file_path = path_to_collection

    def do_import(self) -> typing.List[VanDerAaDocument]:
        documents: typing.Dict[str, VanDerAaDocument] = {}

        file_paths = [self._file_path]
        if os.path.isdir(self._file_path):
            file_paths = os.listdir(self._file_path)
            file_paths = [os.path.join(self._file_path, f) for f in file_paths]

        for file_path in file_paths:
            with open(file_path, "r", encoding="windows-1252") as f:
                reader = csv.reader(f, delimiter=";")
                # strip header
                _ = next(reader)

                for row in reader:
                    file_name = os.path.basename(file_path)
                    file_name, _ = os.path.splitext(file_name)
                    doc_id = row[NAME_COL]
                    text = row[TEXT_COL]

                    # quishpi uses 1 for all rows and files, this would not be unique...
                    doc_id = f"{file_name}-{doc_id}"

                    if doc_id not in documents:
                        documents[doc_id] = VanDerAaDocument(
                            id=doc_id,
                            text="",
                            name=doc_id,
                            constraints=[],
                            sentences=[],
                        )
                    document = documents[doc_id]

                    sentence_index = len(document.sentences)
                    constraints = self.parse_constraints(row, sentence_index)

                    document.sentences.append(text)
                    document.constraints.extend(constraints)
                    document.text += f"\n{text}"

        return list(documents.values())

    @staticmethod
    def parse_constraints(
        row: typing.List, sentence_index: int
    ) -> typing.List[VanDerAaConstraint]:
        max_constraints = 3
        num_constraints = int(row[NUM_CONSTRAINTS_COL])
        base_index = CONSTRAINT_1_COL
        constraints: typing.List[VanDerAaConstraint] = []
        for i in range(num_constraints):
            constraint_type = row[base_index + i * 3 + 0]
            constraint_head = row[base_index + i * 3 + 1]
            constraint_tail = row[base_index + i * 3 + 2]
            constraint_negative = row[NEGATIVE_COL].lower().strip() == "true"

            if constraint_type.strip() == "":
                # no constraint given
                continue

            if constraint_tail == "":
                constraint_tail = None
            constraints.append(
                VanDerAaConstraint(
                    type=constraint_type.strip().lower(),
                    head=constraint_head,
                    tail=constraint_tail,
                    negative=constraint_negative,
                    sentence_id=sentence_index,
                )
            )
        if num_constraints != len(constraints):
            print(
                f"Mismatch between the given number of constraints ({num_constraints}) "
                f"and the actual constraints listed ({len(constraints)}). "
                f"This is not a problem, but indicates the dataset is inconsistent."
            )
        return constraints


if __name__ == "__main__":

    def main():
        documents = VanDerAaImporter(
            "../res/data/van-der-aa/datacollection.csv"
        ).do_import()
        print(len(documents))

        documents = VanDerAaImporter("../res/data/quishpi/csv").do_import()
        print(len(documents))

    main()
