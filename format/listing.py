import re
import typing

import data
from format import base, common, tags


class VanDerAaRelationListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["constraints"]],
        prompt_path: str,
        separate_tasks: bool,
    ):
        super().__init__(steps)
        self._prompt_path = prompt_path
        self._prompt = common.load_prompt_from_file(prompt_path)
        self._separate_tasks = separate_tasks
        self._sentence_re = re.compile(
            r"^\s*\*\*\s?sentence\s*(\d+)\s*\*\*\s*$", flags=re.IGNORECASE
        )

    def description(self) -> str:
        return self._prompt

    @property
    def args(self):
        return {
            "prompt_path": self._prompt_path,
            "separate_tasks": self._separate_tasks,
        }

    def _dump_constraints(
        self, document: data.VanDerAaDocument, sentence_id: int
    ) -> str:
        res = []
        for c in document.constraints:
            if c.sentence_id != sentence_id:
                continue
            negative = "TRUE" if c.negative else "FALSE"
            tail = ""
            if c.tail is not None:
                tail = c.tail
            if self._separate_tasks:
                res.append(f"{negative}\t{c.type}\t{c.head}\t{tail}")
            else:
                res.append(f"{c.sentence_id}\t{negative}\t{c.type}\t{c.head}\t{tail}")
        return "\n".join(res)

    @staticmethod
    def _dump_actions(document: data.VanDerAaDocument, sentence_id: int) -> str:
        actions: typing.Set[str] = set()
        for c in document.constraints:
            if c.sentence_id != sentence_id:
                continue
            actions.add(c.head)
            if c.tail is not None:
                actions.add(c.tail)
        return "\n".join(actions)

    def output(self, document: data.VanDerAaDocument) -> str:
        res = []
        for i, sentence in enumerate(document.sentences):
            if self._separate_tasks:
                res.append(f"** Sentence {i} **")
                res.append("")
                res.append("Actions:")
                res.append(self._dump_actions(document, i))
                res.append("")
                res.append("Constraints:")
            res.append(self._dump_constraints(document, i))
            if self._separate_tasks:
                res.append("")
        return "\n".join(res)

    def input(self, document: data.VanDerAaDocument) -> str:
        return "\n".join(f"Sentence {i}: {s}" for i, s in enumerate(document.sentences))

    def parse(
        self, document: data.VanDerAaDocument, string: str
    ) -> data.VanDerAaDocument:
        constraints = []
        current_sentence_id: typing.Optional[int] = None
        for line in string.splitlines(keepends=False):
            if line.strip() == "":
                continue

            match = re.match(self._sentence_re, line)
            if match is not None:
                # new sentence
                current_sentence_id = int(match.group(1))
                continue

            if "\t" not in line:
                # either a header like "Actions:" or "Constraints:",
                # or an action, which we currently do not parse
                continue

            split_line = line.strip().split("\t")
            if self._separate_tasks:
                if len(split_line) == 4:
                    negative, c_type, c_head, c_tail = split_line
                elif len(split_line) == 3:
                    negative, c_type, c_head = split_line
                    c_tail = None
                else:
                    print(
                        f'Expected 3 or 4 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
            else:
                if len(split_line) == 5:
                    current_sentence_id, negative, c_type, c_head, c_tail = split_line
                    if c_tail == "":
                        c_tail = None
                elif len(split_line) == 4:
                    current_sentence_id, negative, c_type, c_head = split_line
                    c_tail = None
                else:
                    print(
                        f'Expected 4 or 5 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
                current_sentence_id = int(current_sentence_id)

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}. Skipping.")
                continue

            if c_tail == "None":
                raise AssertionError()

            constraints.append(
                data.VanDerAaConstraint(
                    sentence_id=current_sentence_id,
                    type=c_type.strip().lower(),
                    head=c_head,
                    tail=c_tail,
                    negative=negative.lower() == "true",
                )
            )
        return data.VanDerAaDocument(
            id=document.id,
            name=document.name,
            text=document.text,
            constraints=constraints,
            sentences=document.sentences,
        )


class QuishpiMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.QuishpiDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["mentions"]]):
        super().__init__(steps)

    def description(self) -> str:
        return common.load_prompt_from_file("quishpi/md/long-no-explain.txt")

    @property
    def args(self):
        return {}

    def output(self, document: data.QuishpiDocument) -> str:
        mentions = []
        for mention in document.mentions:
            mentions.append(f"{mention.type}\t{mention.text}")
        return "\n".join(mentions)

    def input(self, document: data.QuishpiDocument) -> str:
        return document.text

    def parse(
        self, document: data.QuishpiDocument, string: str
    ) -> data.QuishpiDocument:
        mentions: typing.List[data.QuishpiMention] = []

        for line in string.splitlines(keepends=False):
            if "\t" not in line:
                print(f"Skipping non-tab-separated line '{line}'.")
                continue

            split_line = line.split("\t")
            assert 2 <= len(split_line) <= 3, split_line
            if 3 < len(split_line) < 2:
                print(
                    f"Expected two or three tab-separated values, "
                    f"got {len(split_line)} in '{line}' from LLM. Skipping."
                )
                continue

            if len(split_line) == 3:
                mention_type, mention_text, explanation = split_line
                print(f"Explanation for {mention_text} ({mention_type}): {explanation}")
            else:
                mention_type, mention_text = split_line

            mention = data.QuishpiMention(
                type=mention_type.strip().lower(), text=mention_text
            )
            mentions.append(mention)

        return data.QuishpiDocument(
            id=document.id, text=document.text, mentions=mentions
        )


class PetRelationListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._input_formatter = tags.PetTagFormattingStrategy(include_ids=True)

    def description(self) -> str:
        return common.load_prompt_from_file("pet/re/long.txt")

    @property
    def args(self):
        return {}

    def output(self, document: data.PetDocument) -> str:
        res = []
        for r in document.relations:
            res.append(f"{r.type}\t{r.head_mention_index}\t{r.tail_mention_index}")
        return "\n".join(res)

    def input(self, document: data.PetDocument) -> str:
        return self._input_formatter.output(document)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=["relations"])
        for line in string.splitlines(keepends=False):
            if "\t" not in line:
                print(f"Skipping non-tab-separated line {line}.")
                continue
            split_line = line.split("\t")
            if len(split_line) != 3:
                print(
                    f"Expected exactly 3 arguments in line {line}, got {len(split_line)}. Skipping."
                )
                continue
            relation_type, head_index, tail_index = split_line
            relation_type = relation_type.lower().strip()
            head_index = int(head_index)
            tail_index = int(tail_index)
            document.relations.append(
                data.PetRelation(
                    type=relation_type,
                    head_mention_index=head_index,
                    tail_mention_index=tail_index,
                )
            )
        return document


class PetEntityListingFormattingStrategy(base.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=True, only_tags=["Activity Data", "Actor"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/er/long.txt")

    @property
    def args(self):
        return {}

    def output(self, document: data.PetDocument) -> str:
        ret = []
        for e in document.entities:
            ret.append(" ".join([str(i) for i in e.mention_indices]))
        return "\n".join(ret)

    def input(self, document: data.PetDocument) -> str:
        return self._input_formatter.output(document)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=["entities"])
        for line in string.splitlines(keepends=False):
            if " " not in line:
                try:
                    mention_ids = [int(line)]
                except ValueError:
                    print(f"Skipping non space-separated line '{line}'!")
                    continue
            else:
                mention_ids = [int(i) for i in line.split(" ")]
            mentions = [document.mentions[i] for i in mention_ids]
            mention_types = set(m.type for m in mentions)
            if len(mention_types) > 1:
                print(f"Extracted multi-type entity, with mentions {mentions}.")
            document.entities.append(data.PetEntity(mention_indices=tuple(mention_ids)))
        for i, mention in enumerate(document.mentions):
            if any([i in e.mention_indices for e in document.entities]):
                continue
            document.entities.append(data.PetEntity(mention_indices=(i,)))
        return document


class PetMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    def __init__(
        self,
        steps: typing.List[str],
        only_tags: typing.Optional[typing.List[str]] = None,
        generate_descriptions: bool = False,
        prompt: str = None,
    ):
        super().__init__(steps)
        self._generate_descriptions = generate_descriptions
        self._only_tags = only_tags
        if self._only_tags is not None:
            self._only_tags = [t.lower() for t in self._only_tags]
        self._prompt = prompt

    def description(self) -> str:
        if self._prompt is None:
            return common.load_prompt_from_file("pet/md/short_prompt.tx")
        else:
            return common.load_prompt_from_file(self._prompt)

    @property
    def args(self):
        return {
            "only_tags": self._only_tags,
            "generate_descriptions": self._generate_descriptions,
            "prompt": self._prompt,
        }

    def output(self, document: data.PetDocument) -> str:
        formatted_mentions = []
        for i, m in enumerate(document.mentions):
            if self._only_tags is not None and m.type.lower() not in self._only_tags:
                continue

            relation_candidates = [
                r
                for r in document.relations
                if r.head_mention_index == i or r.tail_mention_index == i
            ]
            relevant_relations = [
                r
                for r in relation_candidates
                if r.type.lower() in ["uses", "actor performer", "actor recipient"]
            ]

            description = ""
            if self._generate_descriptions:
                if len(relevant_relations) > 0:
                    relevant_relation = relevant_relations[0]
                    head = document.mentions[relevant_relation.head_mention_index]
                    tail = document.mentions[relevant_relation.tail_mention_index]
                    if relevant_relation.type.lower() == "uses":
                        description = f'"{tail.text(document)}" is an object that is being used in the activity "{head.text(document)}"'
                    if relevant_relation.type.lower() == "actor performer":
                        description = f'"{tail.text(document)}" is an actor that executes the activity "{head.text(document)}"'
                    if relevant_relation.type.lower() == "actor recipient":
                        description = f'"{tail.text(document)}" is an actor that is directly affected by the activity "{head.text(document)}"'
                if description == "":
                    print(
                        f"WARNING: no description generated for '{m.text(document)}'! "
                        f"{len(relation_candidates)} relevant relations "
                        f"(candidates were: {relation_candidates})"
                    )

            first_token = document.tokens[m.token_document_indices[0]]
            formatted_mentions.append(
                f"{m.text(document)}\t{m.type}\t{first_token.sentence_index}\t{description}"
            )
        return "\n".join(formatted_mentions)

    def input(self, document: data.PetDocument) -> str:
        sentences = document.sentences
        text = ""
        for i, sentence in enumerate(sentences):
            text += f"Sentence {i}: "
            text += " ".join(t.text for t in sentence)
            text += "\n"
        return text

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        sentences = document.sentences
        parsed_mentions: typing.List[data.PetMention] = []
        for line in string.splitlines(keepends=False):
            if re.match("-{3,}", line.strip()):
                print(
                    "Found divider, will discard all mentions, as they were only candidates"
                )
                parsed_mentions = []
                continue

            if "\t" not in line:
                print(f"line not tab-separated: '{line}'")
                continue
            split_line = line.split("\t")
            split_line = tuple(e for e in split_line if e.strip() != "")

            if len(split_line) < 3 or len(split_line) > 4:
                print(
                    f"Skipping line {split_line}, as it is not formatted "
                    f"properly, expected between 3 and 4 arguments."
                )
                continue

            if len(split_line) == 3:
                mention_text, mention_type, sentence_id = split_line
            else:
                mention_text, mention_type, sentence_id, explanation = split_line
                print(f"Explanation for {mention_text}: {explanation}")

            try:
                sentence_id = int(sentence_id)
            except ValueError:
                print(f"Invalid sentence index '{sentence_id}', skipping line.")
                continue
            sentence = sentences[sentence_id]

            mention_text = mention_text.lower()
            mention_tokens = mention_text.split(" ")

            matches_in_sentence = 0
            for i, token in enumerate(sentence):
                candidates = sentence[i : i + len(mention_tokens)]
                candidate_text = " ".join(c.text.lower() for c in candidates)

                if candidate_text.lower() != mention_text.lower():
                    continue

                parsed_mentions.append(
                    data.PetMention(
                        token_document_indices=tuple(
                            c.index_in_document for c in candidates
                        ),
                        type=mention_type.lower().strip(),
                    )
                )
                matches_in_sentence += 1
            if matches_in_sentence == 0:
                print(
                    f"No match for line with parsed sentence id {sentence_id}: '{line}'"
                )
            if matches_in_sentence > 1:
                print(
                    f"Multiple matches for line with parsed sentence id {sentence_id}: '{line}'"
                )
        return data.PetDocument(
            id=document.id,
            text=document.text,
            name=document.name,
            category=document.category,
            tokens=[t.copy() for t in document.tokens],
            mentions=parsed_mentions,
            relations=[],
            entities=[],
        )


class PetActivityListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["activity"], generate_descriptions=False)

    @property
    def args(self):
        return {}

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/activities.txt")


class IterativePetMentionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(
        self, steps: typing.List[str], tag: str, context_tags: typing.List[str]
    ):
        super().__init__(steps, only_tags=[tag], generate_descriptions=False)
        self._tag = tag.lower()
        self._context_tags = [t.lower() for t in context_tags]
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=self._context_tags
        )

    @property
    def args(self):
        return {"tag": self._tag, "context_tags": self._context_tags}

    def description(self) -> str:
        return common.load_prompt_from_file(
            f"pet/md/iterative/no_explanation/{self._tag.replace(' ', '_')}.txt"
        )

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetActorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["actor"], generate_descriptions=False)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file(
            "pet/md/iterative/actors_no_explanation.txt"
        )

    @property
    def args(self):
        return {}

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetAndListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["and gateway"], generate_descriptions=False)

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/and.txt")

    @property
    def args(self):
        return {}


class PetConditionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["condition specification"], generate_descriptions=False
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/condition.txt")

    @property
    def args(self):
        return {}


class PetDataListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["activity data"], generate_descriptions=False
        )
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity", "Actor"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/data_no_explanation.txt")

    @property
    def args(self):
        return {}

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetFurtherListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["further specification"], generate_descriptions=False
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/further.txt")

    @property
    def args(self):
        return {}


class PetXorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["xor gateway"], generate_descriptions=False)

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/xor.txt")

    @property
    def args(self):
        return {}


if __name__ == "__main__":

    def main():
        documents = data.PetImporter("../res/data/pet/all.new.jsonl").do_import()
        formatter = PetMentionListingFormattingStrategy(steps=["mentions"])
        for d in documents:
            print("Input:")
            print(formatter.input(d))
            print()
            print("Output:")
            print(formatter.output(d))
            print()
            print("-------")
            print()
            formatter.parse(d, formatter.output(d))

    main()
