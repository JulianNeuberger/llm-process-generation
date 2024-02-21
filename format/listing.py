import data
from format import base

prompt = """You are a business process modelling expert, tasked with identifying
constraints between actions in textual process descriptions. Textual process
descriptions are sentences that describe a short sequence of actions. Ordering and existence of actions depend on
constraints between them. Below you find further details about actions and constraints:

- action: predicate and object describing a task. Predicate is usually a verb, and object is 
          some physical or digital object on which is being acted on. 

- constraint: defines if and how actions can be executed. Always has a source / head 
              action and sometimes a target / tail action, depending on the type. All 
              constraints are one of the following types:
    - init: this constraint marks a single action as the start of a work process. It has 
            no target / tail action, only a source / head. 
    - end: this constraint marks a single action as the end of a work process. It has 
           no target / tail action, only a source / head.
    - precedence: The tail action can only be executed, if the head was already executed
                  before. the head may be executed without the tail being executed.
    - response: if the head action was executed, the tail action also has to be executed,
                but the tail can also be executed independently of head. 
    - succession: if the head action was executed, the tail is also executed, neither of
                  them are executed in isolation, i.e., tail is always executed after head,
                  or none of them are executed at all.

Additionally you can determine if the given document describes a negation of constraints, 
e.g., "when something happens, then we DO something" describes a positive constraint,
while "when something happens, then we DON'T DO something" describes a negation.

Note that constraints having a tail and a head are usually formulated like condition-consequence pairs. They restrict 
different situations in the execution of process by describing the situation in the form of a condition and the
consequence as kind of an implication.

Please extract all constraints in the given raw text in the following format:
Print one constraint per line, where you separate if the constraint is negative (TRUE if 
the document describes a negation, else it reads FALSE), the type of the constraint, and the 
extracted actions by tabs, e.g. "TRUE\tsuccession\tdo something\tdo thing". An example for the 
format would be:

FALSE\tsuccession\teat apple\tthrow ball
TRUE\tprecedence\tplay game\twrite message
TRUE\tresponse\treceive file\tsend results

Actions should be made up of predicate and object, where the predicate is a verb in infinitive and the object
is usually a noun or a pronoun. Determiners are usually NOT part of an action.

Please return raw text, do not use any formatting.
"""


class VanDerAaListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    @staticmethod
    def description() -> str:
        return prompt

    def output(self, document: data.VanDerAaDocument) -> str:
        constraints = []
        negative = "TRUE" if document.negative else "FALSE"
        for constraint in document.constraints:
            constraints.append(
                f"{constraint.type}\t{constraint.head}\t{constraint.tail}"
            )
        formatted_constraints = "\n".join(constraints)
        return f"{negative}\n{formatted_constraints}"

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
