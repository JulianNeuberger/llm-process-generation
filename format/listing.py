import typing

import data
from format import base, common, tags


class VanDerAaListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("van-der-aa/re/long.txt")

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


class PetEntityListingFormattingStrategy(base.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=True, only_tags=["Activity Data", "Actor"]
        )

    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/er/long.txt")

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

quishpi_re_prompt = """You are a business process modelling expert, tasked with identifying
constraints between actions in textual process descriptions. Processes consist of actions and, thus, textual process
descriptions are sentences that describe a short sequence of actions. Ordering and existence of actions depend on
constraints between them. Below you find further details about actions and constraints:

- action: predicate and object describing a task. Predicate is usually a transitive verb, and object is 
          some physical or digital object on which is being acted on. 

- constraint: defines if and how actions can be executed. Always has a source / head 
              action and sometimes a target / tail action, depending on the type. All 
              constraints are one of the following types:
    - init: marks an action as the start of an entire process. This action is the source / head action of the init 
    constraint. There is no target / tail action. Note that it must be explicitly stated that the PROCESS is started 
    for an init constraint to apply. Signal words alone are not sufficient here.  
    - end: marks an action as the end of the whole process. The action is the source / head action. There is no
            no target / tail action. Note that it must be explicitly stated that the PROCESS is ended 
    for an end constraint to apply. Signal words alone are not sufficient here. 
    - precedence: The tail action can only be executed, if the head was already executed
                  before. the head may be executed without the tail being executed.
    - response: if the head action was executed, the tail action has to be executed, too.
    - succession: this means if the head activity is executed, the tail activity needs to be
          executed as well and at the same time, the tail activity requires prior execution of the head activity. 
    - existence: requires that an action is executed at some point in the process; the execution is not dependent on 
        any other explicitly mentioned action; there is only a head action but no tail action.
    - absence: requires that an action is NOT executed at any time in the process; the absence is not dependent on 
        any other action but other circumstances (e.g., information from the process context); there is only a head 
        action but no tail action; Note: negated absence constraint is semantically equivalent to an existence 
        constraint.
    - noncooccurrence: requires the head action is not executed if the tail action is executed and vice versa.

Additionally you can determine if the given document describes a negation of constraints, 
e.g., "when something happens, then we DO something" describes a positive constraint,
while "when something happens, then we DON'T DO something" describes a negation.

Please extract all constraints in the given raw text in the following format:
Print one constraint per line, where you separate if the constraint is negative (TRUE if 
the document describes a negation, else it reads FALSE), the type of the constraint, and the 
extracted actions by tabs in the following form (<...> are placeholders): 
<TRUE or FALSE>\t<constraint type>\t<head action>\t<tail action>. 

Please return raw text, do not use any formatting.
"""


class QuishpiREListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return quishpi_re_prompt

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
