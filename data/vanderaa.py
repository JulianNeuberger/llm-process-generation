import csv
import dataclasses
import difflib
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


@dataclasses.dataclass(eq=True, frozen=True)
class VanDerAaMention(base.SupportsPrettyDump["VanDerAaDocument"], base.HasCustomMatch):
    text: str
    sentence_id: int
    mandatory: bool

    def pretty_dump(self, document: TDocument, human_readable: bool = False) -> str:
        return self.text

    def copy(self):
        return VanDerAaMention(
            text=self.text, sentence_id=self.sentence_id, mandatory=self.mandatory
        )

    @staticmethod
    def is_action_mandatory(
        action: str, text: str, other_actions: typing.List[str]
    ) -> bool:
        verb = VanDerAaMention.get_action_verb(action)
        tokenized_text = text.lower().split(" ")
        close_matches = difflib.get_close_matches(
            verb.lower(), tokenized_text, cutoff=0.0
        )
        most_likely_match: str = close_matches[0]
        action_index = tokenized_text.index(most_likely_match)

        tokenized_text = tokenized_text[max(0, action_index - 10) : action_index]

        # other_action_indices = []
        #
        # for other_action in other_actions:
        #     verb = VanDerAaMention.get_action_verb(other_action)
        #     close_matches = difflib.get_close_matches(verb, tokenized_text, cutoff=0.0)
        #     most_likely_match: str = close_matches[0]
        #     most_likely_position = tokenized_text.index(most_likely_match)
        #     other_action_indices.append(most_likely_position)
        #
        # other_action_indices.sort()
        # other_action_indices = [i for i in other_action_indices if i < action_index]
        # tokenized_text = tokenized_text[0:action_index]
        # if len(other_action_indices) > 0:
        #     tokenized_text = tokenized_text[other_action_indices[-1] :]

        # check for trigger words that force mandatory
        for t in tokenized_text:
            if t in ["must", "will", "would", "shall", "should", "require", "have to"]:
                return True

        # check for conditions
        for t in tokenized_text:
            if t in [
                "if",
                "once",
                "condition",
                "after",
                "before",
                "when",
                "unless",
                "can",
                "could",
                "may",
                "might",
                "first",
                "before",
                "earlier",
                "after",
                "later",
            ]:
                return False

        # by default everything else is mandatory
        return True

    @staticmethod
    def get_action_verb(action_text: str):
        tokens = nltk.word_tokenize(action_text)
        pos_tags: typing.Iterable[typing.Tuple[str, str]] = nltk.pos_tag(tokens)
        pred_verb: typing.Optional[str] = None
        for token, pos in pos_tags:
            if pos.startswith("VB"):
                pred_verb = token
                break

        if pred_verb is None:
            pred_verb = tokens[0]
        return pred_verb

    def match(self, other: object) -> bool:
        if not isinstance(other, VanDerAaMention):
            return False

        pred = other.text.lower()
        true = self.text.lower()

        true_verb = true.split(" ")[0]
        if true_verb.lower() in pred:
            return True
        pred_verb = self.get_action_verb(pred)

        if pred_verb.lower() in true:
            return True
        return False


@dataclasses.dataclass
class VanDerAaDocument(base.DocumentBase):
    name: str
    sentences: typing.List[str]
    constraints: typing.List["VanDerAaConstraint"]
    mentions: typing.List[VanDerAaMention]

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
        mentions = []
        if "mentions" not in clear:
            mentions = [c.copy() for c in self.mentions]
        return VanDerAaDocument(
            id=self.id,
            text=self.text,
            name=self.name,
            constraints=constraints,
            sentences=self.sentences,
            mentions=mentions,
        )


@dataclasses.dataclass(eq=True, frozen=True)
class VanDerAaConstraint(
    base.SupportsPrettyDump["VanDerAaDocument"], base.HasCustomMatch, base.HasType
):
    head: VanDerAaMention
    tail: typing.Optional[VanDerAaMention]
    negative: bool
    sentence_id: int

    def pretty_dump(
        self, document: VanDerAaDocument, human_readable: bool = False
    ) -> str:
        separator = ";\t" if human_readable else "\t"
        pretty = f'{"TRUE" if self.negative else "FALSE"}{separator}{self.type}{separator}{self.head.pretty_dump(document, human_readable)}'
        if self.tail:
            pretty = (
                f"{pretty}{separator}{self.tail.pretty_dump(document, human_readable)}"
            )
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

    def correct_slots(self, true: "VanDerAaConstraint") -> int:
        res = 0
        if self.type.lower() == true.type.lower():
            res += 1
        if true.head.match(self.head):
            res += 1
        if true.tail is not None:
            if true.tail.match(self.tail):
                res += 1
        if self.negative and true.negative:
            res += 1
            if self.type.lower() != true.type.lower():
                res += 1
        return res

    def match(self, other: object) -> bool:
        if not isinstance(other, VanDerAaConstraint):
            return False
        if not self.head.match(other.head):
            return False
        if self.tail is None and other.tail is not None:
            return False
        if self.tail is None and other.tail is None:
            return self.type.lower() == other.type.lower()
        if not self.tail.match(other.tail):
            return False
        if self.negative and not other.negative:
            return False
        if self.sentence_id != other.sentence_id:
            return False
        return self.type.lower() == other.type.lower()


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
                            mentions=[],
                        )
                    document = documents[doc_id]

                    sentence_index = len(document.sentences)
                    constraints = self.parse_constraints(row, sentence_index)

                    document.sentences.append(text)
                    document.constraints.extend(constraints)

                    new_actions = []
                    for c in constraints:
                        if (
                            c.head not in document.mentions
                            and c.head is not None
                            and c.head not in new_actions
                        ):
                            new_actions.append(c.head)
                        if (
                            c.tail not in document.mentions
                            and c.tail is not None
                            and c.tail not in new_actions
                        ):
                            new_actions.append(c.tail)

                    # fix mandatory state of actions
                    new_action_texts = [m.text for m in new_actions]
                    for i, m in enumerate(new_actions):
                        new_actions[i] = VanDerAaMention(
                            text=m.text,
                            sentence_id=m.sentence_id,
                            mandatory=VanDerAaMention.is_action_mandatory(
                                m.text, text, new_action_texts
                            ),
                        )

                    document.mentions.extend(new_actions)

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

            constraint_head = VanDerAaMention(
                text=constraint_head,
                sentence_id=sentence_index,
                mandatory=False,
            )
            if constraint_tail == "":
                constraint_tail = None
            else:
                constraint_tail = VanDerAaMention(
                    text=constraint_tail,
                    sentence_id=sentence_index,
                    mandatory=False,
                )
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

        by_tag: typing.Dict[
            str,
            typing.List[typing.Tuple[VanDerAaDocument, VanDerAaConstraint]],
        ] = {}
        for d in documents:
            for c in d.constraints:
                if c.type not in by_tag:
                    by_tag[c.type] = []
                by_tag[c.type].append((d, c))
        for tag, constraints in by_tag.items():
            print(tag)
            print(
                "-----------------------------------------------------------------------"
            )
            for d, c in constraints:
                print(c.pretty_dump(d, True) + "\t\t" + d.sentences[c.sentence_id])
            print(
                "-----------------------------------------------------------------------"
            )
        for d in documents:
            for m in d.mentions:
                print(m)

    main()
