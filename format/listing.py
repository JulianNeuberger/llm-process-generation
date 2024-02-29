import typing

import data
from format import base, common

van_der_aa_prompt = common.load_prompt_from_file("van-der-aa/")


class VanDerAaListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["constraints"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return van_der_aa_prompt

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


quishpi_prompt = common.load_prompt_from_file("quishpi/md/long.txt")


class QuishpiListingFormattingStrategy(
    base.BaseFormattingStrategy[data.QuishpiDocument]
):
    def __init__(self, steps: typing.List[typing.Literal["mentions"]]):
        super().__init__(steps)

    @staticmethod
    def description() -> str:
        return quishpi_prompt

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
            split_line = line.split("\t")
            if len(split_line) != 2:
                print(
                    f"Expected two tab-separated values, got {len(split_line)} in '{line}' from LLM."
                )
                continue
            mention_type, mention_text = split_line
            mention = data.QuishpiMention(
                type=mention_type.strip().lower(), text=mention_text
            )
            mentions.append(mention)

        return data.QuishpiDocument(
            id=document.id, text=document.text, mentions=mentions
        )


generated_pet_prompt = common.load_prompt_from_file("pet/md/generated_prompt.txt")
pet_prompt = common.load_prompt_from_file("pet/md/long_prompt.txt")
short_pet_prompt = common.load_prompt_from_file("pet/md/short_prompt.tx")


class PetMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    @staticmethod
    def description() -> str:
        return short_pet_prompt

    def output(self, document: data.PetDocument) -> str:
        formatted_mentions = []
        for m in document.mentions:
            first_token = document.tokens[m.token_document_indices[0]]
            formatted_mentions.append(
                f"{m.text(document)}\t{m.type}\t{first_token.sentence_index}"
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
                continue
            split_line = line.split("\t")
            split_line = tuple(e for e in split_line if e.strip() != "")
            assert len(split_line) == 3, split_line
            mention_text, mention_type, sentence_id = split_line
            sentence_id = int(sentence_id)
            sentence = sentences[sentence_id]
            mention_tokens = mention_text.split(" ")
            for i, token in enumerate(sentence):
                candidates = sentence[i : i + len(mention_tokens)]
                candidate_text = " ".join(c.text for c in candidates)

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
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/activities.txt")


class PetActorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/actors.txt")


class PetAndListingFormattingStrategy(PetMentionListingFormattingStrategy):
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/and.txt")


class PetConditionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/condition.txt")


class PetDataListingFormattingStrategy(PetMentionListingFormattingStrategy):
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/data.txt")


class PetFurtherListingFormattingStrategy(PetMentionListingFormattingStrategy):
    @staticmethod
    def description() -> str:
        return common.load_prompt_from_file("pet/md/iterative/further.txt")


class PetXorListingFormattingStrategy(PetMentionListingFormattingStrategy):
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
