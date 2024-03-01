import typing

import data
from format import base, common, tags

from format.prompts import quishpi_re_prompt, vanderaa_prompt


class VanDerAaListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return vanderaa_prompt.VAN_DER_AA_PROMPT

    def output(self, document: data.VanDerAaDocument) -> str:
        constraints = []
        for constraint in document.constraints:
            constraints.append(
                f"{constraint.type}\t{constraint.head}\t{constraint.tail}"
            )
        return "\n".join(constraints)

    def input(self, document: data.VanDerAaDocument) -> str:
        return document.text

    def parse(
        self, document: data.VanDerAaDocument, string: str
    ) -> data.VanDerAaDocument:
        if "#-#-#RESULT#-#-#" in string:
            string = string.split("#-#-#RESULT#-#-#")[1].strip()
        lines = string.splitlines(keepends=False)
        constraints = []
        for line in lines:
            split_line = line.strip().split("\t")
            if len(split_line) == 4:
                negative, c_type, c_head, c_tail = split_line
            elif len(split_line) == 3:
                negative, c_type, c_head = split_line
                c_tail = None
            else:
                print(
                    f'Expected 2-3 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                )
                continue

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}")
            constraints.append(
                data.VanDerAaConstraint(
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
        )


class VanDerAaStepwiseListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return vanderaa_prompt.VAN_DER_AA_PROMPT_STEPWISE

    def output(self, document: data.VanDerAaDocument) -> str:
        constraints = []
        for constraint in document.constraints:
            constraints.append(
                f"{constraint.type}\t{constraint.head}\t{constraint.tail}"
            )
        return "\n".join(constraints)

    def input(self, document: data.VanDerAaDocument) -> str:
        return document.text

    def parse(
        self, document: data.VanDerAaDocument, string: str
    ) -> data.VanDerAaDocument:
        lines = string.splitlines(keepends=False)
        constraints = []
        for line in lines:
            split_line = line.strip().split("\t")
            if len(split_line) == 4:
                negative, c_type, c_head, c_tail = split_line
            elif len(split_line) == 3:
                negative, c_type, c_head = split_line
                c_tail = None
            else:
                print(
                    f'Expected 2-3 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                )
                continue

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}")
            constraints.append(
                data.VanDerAaConstraint(
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
        )


class QuishpiREListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return quishpi_re_prompt.QUISHPI_RE_PROMPT_HANDCRAFTED_TASK_SEPARATION

    def output(self, document: data.VanDerAaDocument) -> str:
        constraints = []
        for constraint in document.constraints:
            constraints.append(
                f"{'TRUE' if constraint.negative else 'FALSE'}\t{constraint.type}\t{constraint.head}\t{constraint.tail}"
            )
        return "\n".join(constraints)

    def input(self, document: data.VanDerAaDocument) -> str:
        return document.text

    def parse(
        self, document: data.VanDerAaDocument, string: str
    ) -> data.VanDerAaDocument:
        if "#-#-#RESULT#-#-#" in string:
            string = string.split("#-#-#RESULT#-#-#")[1].strip()
        lines = string.splitlines(keepends=False)
        constraints = []
        for line in lines:
            split_line = line.strip().split("\t")
            if len(split_line) == 4:
                negative, c_type, c_head, c_tail = split_line
            elif len(split_line) == 3:
                negative, c_type, c_head = split_line
                c_tail = None
            else:
                print(
                    f'Expected 2-3 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                )
                continue

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}")
                continue
            constraints.append(
                data.VanDerAaConstraint(
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
        )


class QuishpiListingFormattingStrategy(
    base.BaseFormattingStrategy[data.QuishpiDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["mentions"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("quishpi/md/long-no-explain.txt")

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


class PetMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    def __init__(
        self,
        steps: typing.List[str],
        only_tags: typing.Optional[typing.List[str]] = None,
        generate_descriptions: bool = False,
    ):
        super().__init__(steps)
        self._generate_descriptions = generate_descriptions
        self._only_tags = only_tags
        if self._only_tags is not None:
            self._only_tags = [t.lower() for t in self._only_tags]

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/short_prompt.tx")

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
            if "\t" not in line:
                print(f"line not tab-separated: '{line}'")
                continue
            split_line = line.split("\t")
            split_line = tuple(e for e in split_line if e.strip() != "")

            assert 3 <= len(split_line) <= 4, split_line

            if len(split_line) == 3:
                mention_text, mention_type, sentence_id = split_line
            else:
                mention_text, mention_type, sentence_id, explanation = split_line
                print(f"Explanation for {mention_text}: {explanation}")

            sentence_id = int(sentence_id)
            sentence = sentences[sentence_id]

            mention_text = mention_text.lower()
            mention_tokens = mention_text.split(" ")

            matches_in_sentence = 0
            for i, token in enumerate(sentence):
                candidates = sentence[i : i + len(mention_tokens)]
                candidate_text = " ".join(c.text.lower() for c in candidates)

                if candidate_text != mention_text:
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

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/activities.txt")


class PetActorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["actor"], generate_descriptions=True)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity"]
        )

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/actors.txt")

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

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/and.txt")


class PetConditionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["condition specification"], generate_descriptions=False
        )

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/condition.txt")


class PetDataListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["activity data"], generate_descriptions=True)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity", "Actor"]
        )

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/data.txt")

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

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/further.txt")


class PetXorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["xor gateway"], generate_descriptions=False)

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/xor.txt")


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
