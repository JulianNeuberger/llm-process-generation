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


generated_pet_prompt = """
As a business process modeling expert, your task is to identify and extract specific elements from textual descriptions of business processes. These elements are critical for creating formal business process models using Declare or BPMN. Focus on the following types of mentions within the text:

- **Actor**: Extract mentions of persons, departments, or roles actively participating in the process. Extract only if the sentence describes the actor performing a task. Include determiners (e.g., "the student") and pronouns ("he", "I", "she"). Example: In "The manager approves the request," extract "The manager" as an Actor.

- **Activity**: Identify active tasks or actions executed during the process. Extract only the verb indicating the action or event (e.g., "approve", "submit"). Do not include the actor performing it or any auxiliary verbs. Example: From "The employee submits the report," extract "submits" as an Activity.

- **Activity Data**: Look for physical objects or digital data relevant to the process because an action produces or uses it. Always include the determiner (e.g., "the form", "a report"). Pronouns like "it" are also considered Activity Data if part of a task description. Example: In "The clerk archives the document," extract "the document" as Activity Data.

- **Further Specification**: Extract information that details how an activity is executed, including means, manner, or conditions. It follows an Activity in the same sentence. Example: If the sentence is "She reviews the application thoroughly," extract "thoroughly" as Further Specification.

- **XOR Gateway**: Identify decision points in the process, usually indicated by "if", "otherwise", "when". Example: In "If the application is complete, proceed to evaluation," extract "If" as an XOR Gateway.

- **AND Gateway**: Spot descriptions of parallel work streams, marked by "while", "meanwhile", "at the same time". Example: "The technician repairs the device while the assistant updates the records," here, "while" is an AND Gateway.

- **Condition Specification**: Defines the condition of an XOR Gateway path, usually following the gateway trigger word. Example: In "If the temperature is above 100, stop the machine," extract "the temperature is above 100" as Condition Specification.

For each mention you detect, write a line in this format: text\ttype\tsentence

- **text**: The exact text of the mention.
- **type**: The type from the ones listed above.
- **sentence**: An integer identifying the sentence where the mention was found, starting from zero.

**Examples**

Given the input text:

Sentence 0: The manager reviews the application.
Sentence 1: If approved, the application proceeds to the next step.

The correct output should be:

The manager\tActor\t0
reviews\tActivity\t0
the application\tActivity Data\t0
If\tXOR Gateway\t1
approved\tCondition Specification\t1

**Note**: Do not alter the extracted text (e.g., correcting typos or changing punctuation). Focus on the essence of the task descriptions and the roles involved without assuming additional context not provided in the text."""

pet_prompt = """
You are a business process modelling expert, tasked with identifying mentions of
process relevant elements in textual descriptions of business processes. These mentions are 
spans of text, that are of a certain type, as described below: 

- **Actor**: a person, department, or similar role that participates actively in the business 
         process, e.g., "the student", "the professor", "the judge", "the clerk". It should
         only be extracted, if the current sentence describes the actor executing a task. 
         Include the determiner if it is present, e.g. extract "the student" from "First the 
         student studies for their exam". Can also be a pronoun, such as "he", "I", "she".
- **Activity**: an active task or action executed during the business process, e.g., 
            "examine", "write", "bake", "review". Do not extract, if it is information about
            the process itself, such as "the process ends", as this is not a task in the process!
            Can also be an event during the process, that is executed by an external, implicit actor, 
            that is not mentioned in the text, e.g., "placed" in "when an order is placed".
            Never contains the Actor that executes it, nor the Activity Data that's used during this
            Activity, is just the verb as in "checked" and not "is checked"! 
- **Activity Data**: a physical object, or digital data that is relevant to the process, because 
                 an action produces or uses it, e.g., "the paper", "the report", "the machine part".
                 Always include the determiner! Can also be a pronoun, like "it", but is always part of 
                 a task description. Is never information about the process itself, as in "the process 
                 ends", here "process" is not Activity Data!
- **Further Specification**: important information about an activity, such as the mean, the manner 
                         of execution, or how an activity is executed. Follows an Activity in the
                         same sentence, and describes how an Activity (task) is being done.
- **XOR Gateway**: textual representation of a decision in the process, usually triggered by adverbs or 
               conjunctions, e.g., "if", "otherwise", "when"
- **AND Gateway**: textual description of parallel work streams in the process, i.e., simultaneous 
               actions performed, marked by phrases like, e.g., "while", "meanwhile", "at the same time" 
- **Condition Specification**: defines the condition of an XOR Gateway path usually directly following the
                           gateway trigger word, e.g., "it is ready", "the claim is valid",
                           "temperature is above 180"

For each mention you detect, write a line in the following format:
text\ttype\tsentence

- **text**: the text of the mention
- **type**: the type from the ones listed above
- **sentence**: integer, that identifies the sentence where the mention text was found in the input. Zero based.
     
**Examples for the format given the input text:** 

*Input*: 

Sentence 0: the professor grades all student papers . 
Sentence 1: some time passes . 
Sentence 2: then the professor returns the papers .

*Output*:

the professor\tActor\t0
grades\tActivity\t0
all student papers\tActivity Data\t0
the professor\tActor\t2
returns\tActivity\t2
the papers\tActivity Data\t2

Do not change the text you extract, i.e., do not correct typos, or change spaces, or add punctuation.
Do not use any code formatting.
"""


class PetMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    @staticmethod
    def description() -> str:
        return generated_pet_prompt

    def output(self, document: data.PetDocument) -> str:
        formatted_mentions = []
        for m in document.mentions:
            formatted_mentions.append(
                f"{m.text(document)}\t{m.type}\t{document.tokens[m.token_document_indices[0]].sentence_index}"
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
        print(document.text)
        print(len(document.sentences))
        print(string)
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
                        token_document_indices=[
                            c.index_in_document for c in candidates
                        ],
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
