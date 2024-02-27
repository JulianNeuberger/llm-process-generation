import typing

import data
from format import base

van_der_aa_prompt = """You are a business process modelling expert, tasked with identifying
constraints between actions in textual process descriptions. Textual process
descriptions are sentences that describe a short sequence of actions and execution 
constraints between them. Below you find a short description of relevant
elements:

- action: predicate and object describing a task. Predicate is usually a verb in 
          infinitive, and object is some physical or digital object on which is
          being acted on.
          
- constraint: defines if and how actions can be executed. Always has a source / head 
              action and sometimes a target / tail action, depending on the type. All 
              constraints are one of the following types:
    - init: this constraint marks a single action as the start of a work process. It has 
            no target / tail action, only a source / head.  
    - end: this constraint marks a single action as the end of a work process. It has 
           no target / tail action, only a source / head.
    - precedence: marks the source / head action as a requirement for the target / tail 
                  action. The tail action can only be executed, if the head was executed
                  first. the head may be executed without the tail being executed.
    - response: if the head action was executed, the tail action also has to be executed,
                but the tail can also be executed independently of head.
    - succession: if the head action was executed, the tail is also executed, neither of
                  them are executed in isolation, i.e., tail is always executed after head,
                  or none of them are executed at all.
                  
Additionally you can determine if the given document describes a negation of constraints, 
e.g., "when something happens, then we do something" describes a positive constraint,
while "if something does not happen, then we do something" describes a negation.

Please extract all constraints in the given raw text in the following format:
Print one constraint per line, where you separate if the constraint is negative (TRUE if 
the document describes a negation, else it reads FALSE), the type of the constraint, and the 
extracted actions by tabs, e.g. "TRUE\tsuccession\tdo something\tdo thing". An example for the 
format would be:

FALSE\tsuccession\tdo something\tdo thing
TRUE\tprecedence\tdo thing\treact to thing
TRUE\tresponse\treceive ok\tsend something

Actions should be made up of predicate and object, where the predicate is a verb in infinitive. 

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


quishpi_re_prompt = """You are a business process modelling expert, tasked with identifying
constraints between actions in textual process descriptions. Processes consist of actions and, thus, textual process
descriptions are sentences that describe a short sequence of actions. Ordering and existence of actions depend on
constraints between them. Below you find further details about actions and constraints:

- action: predicate and object describing a task. Predicate is usually a verb, and object is 
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

Note that constraints having a tail and a head are usually formulated like condition-consequence pairs. They restrict 
different situations in the execution of process by describing the situation in the form of a condition and the
consequence as kind of an implication. Constraints are ALWAYS described explicitly. Please do NEVER try to guess 
the valid constraints from the context and your own interpretation of the process. Stick closely to what is written in
the process description. Further note that response, precedence, existence and succession are easy to be mixed up. 
However, it is easy to disambiguate response, precedence and succession if you carefully consider the modality if 
an action is a precondition for another action (precedence), requires that something else must be done because of the 
action (response) or if it means both (succession). Here are some examples that are NOT a succession constraint:
          1. After doing A, you have to do B. (response)
          2. After doing A, B can be done, too. (precedence)
          3. Action B must be carried out at some point in between. (existence)

Actions should be made up of predicate and object, where the predicate is a verb in infinitive and the object
is usually a noun or a pronoun. Determiners are usually NOT part of an action.

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
