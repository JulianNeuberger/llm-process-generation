import typing

import data
from format import base

van_der_aa_prompt = """Your task is to extract declarative process models from natural language process descriptions. 
The process descriptions consist of a series of actions, each described by a predicate and an object. 
Your goal is to identify constraints between these actions, which dictate the ordering and existence of actions within
the process. Constraints can be one of the following types:

init: Marks an action as the start of the entire process. It has no target action.
end: Marks an action as the end of the whole process. It has no target action.
precedence: Specifies that the tail action can only be executed if the head action was already executed before.
response: Requires that if the head action was executed, the tail action must also be executed.
succession: Specifies that if the head action is executed, the tail action needs to be executed as well, and vice versa.

Additionally, you may encounter negations of constraints, indicated by statements like "do not" or "must not."

Please ensure the correct identification and formatting of constraints in the given text. Output one constraint 
per line, including whether the constraint is negated (TRUE or FALSE), the type of constraint, and the extracted 
head and tail actions separated by tabs. Stick closely to the provided examples and descriptions, and be careful 
to distinguish between precedence, response, and succession constraints.

Here are some examples for both input and expected output (separated by the following symbol: |):
    - Example 1: The process begins when the author submits the paper.|FALSE\tinit\tsubmit paper
    - Example 2: After signing the contract, the product can be advertised.|FALSE\tprecedence\tsign contract\tadvertise product
    - Example 3: After signing the contract, the product is advertised but never before.|FALSE\tsuccession\tsign contract\tadvertise product
    - Example 4: When the manager is called, the request needs to be forwarded to the secretary, too.|FALSE\tresponse\tcall manager\tforward request
    - Example 5: The process is completed as soon as the proposal is archived|FALSE\tend\tarchive proposal
    - Example 6: After notifying the manager, the request must not be rejected.|TRUE\tresponse\tnofify manager\treject request 

Please return raw text, do not use any formatting.
"""


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
            constraints.append(
                data.VanDerAaConstraint(
                    type=c_type.lower(),
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


quishpi_prompt = """You are a business process modelling expert, tasked with finding
actions and conditions in textual business process descriptions. These process descriptions
are natural language texts that define how a business process has to be executed, with 
actions that have to be taken and conditions, that make the execution of some actions 
conditional. More details on actions and conditions:

- action: the task a participant of the process has to execute, a single 
          verb that defines a unit of work. Should not include the object the work is
          executed on / with, nor the participant that executes the action, nor additional 
          specifications, like adjectives or adverbs. Information 
          about the process itself, such as "the process starts" do not constitute an action 
          and should not be extracted. 
- condition: a phrase (span of text) that declares a decision and starts a conditional 
             path in the process, usually the part directly following conditional words 
             e.g., "if", "either", and similar words. The condition does not include the 
             trigger word itself. If the condition does exist only of the trigger word, 
             do not extract it. Only extract, if the condition affects an earlier or 
             following action.

Please extract all mentions in the given raw text in the following format:
Print one mention per line, where you separate the mentions type and text by tabs, 
e.g. "action\tdo something". An example for the 
format would be:

action\tannotate
condition\tit is a relevant element
action\tformatted

Please return raw text, do not use any code formatting. Do not change the text of extracted
actions and conditions, keep it exactly the same as it appears in text.
"""


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
            mention = data.QuishpiMention(type=mention_type, text=mention_text)
            mentions.append(mention)

        return data.QuishpiDocument(
            id=document.id, text=document.text, mentions=mentions
        )
