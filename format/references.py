import typing

import data
import format


prompt = """You are a business process modelling expert, tasked with identifying 
process relevant information in textual descriptions of business processes. Relevant 
information are the following elements: 

- mentions: business process relevant elements, including
    - Actor: a person, department, or similar role that participates actively in the business 
             process, e.g., "the student", "the professor", "the judge", "the clerk"  
    - Activity: a task or action executed by an actor during the business process, e.g., 
                "examine", "write", "bake", "review"
    - Activity Data: a physical object, or digital data that is relevant to the process, because 
                     an action produces or uses it, e.g., "the paper", "the report", "the machine part"
    - Further Specification: important information about an activity, such as the mean, the manner 
                             of execution, or how an activity is executed.
    - XOR Gateway: textual representation of a decision in the process, usually triggered by adverbs or 
                   conjunctions, e.g., "if", "otherwise", "when"
    - AND Gateway: textual description of parallel work streams in the process, i.e., simultaneous 
                   actions performed, marked by phrases like, e.g., "while", "meanwhile", "at the same time" 
    - Condition Specification: defines the condition of an XOR-gateway path usually following the
                               gateway trigger word, e.g., "it is ready", "the claim is valid",
                               "temperature is above 180"
 
- entities: mentions referring to the same process element are part of the same entity. The only
            process elements that can be part of an entity are actors, and activity data. In the 
            text entities can often be identified by pronouns that refer to earlier mentions, or 
            similar texts. Examples for entities are "student", "he"; or "machine part", "it", "part".

- relations: semantic relations among mentions. You are able to extract the following relations:
    - uses: links elements of type "activity data" to those of type "activity", when the former is used during execution of the latter.
    - flow: links elements of type "activity", "condition specification", "xor gateway", "and gateway", if they are executed in order in the process. 
    - actor recipient: defines the actor that receives the result of some activity execution
    - same gateway: links elements of type "xor gateway" and "and gateway", if they describe the same textual descriptions of gateways are often spread over several phrases, especially the different outgoing paths, this relation links these disjointed descriptions together.
    - further specification: links additional important information (mentions of type further specification) about an activity element.

You are given a text with xml style tags that mark mentions of process relevant elements. 
These tags include the element's type and an identifier, which you can use to refer to it. 

Retrieve an entity by listing all the mention indices that are part of it, 
one entity per line, e.g.:

entities:
- indices: 0, 1, 6
- indices: 3, 4

Please only retrieve entities containing at least one mention.

Retrieve a relation by listing the source mention index, target mention 
index and its type, one relation per line, e.g.:

relations:
- headIndex: 3
  tailIndex: 0
  tag: uses
- headIndex: 7
  tailIndex: 6
  tag: flow

The user may ask you to extract entities, relations, or both.

Put out raw text, without code formatting."""


class PetReferencesFormattingStrategy(format.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[typing.Literal["entities", "relations"]]):
        super().__init__(steps)
        self._input_formatter = format.PetTagFormattingStrategy(include_ids=True)
        self._output_formatter = format.PetYamlFormattingStrategy(steps=steps)

    def description(self) -> str:
        return prompt

    def output(self, document: data.PetDocument) -> str:
        return self._output_formatter.output(document)

    def input(self, document: data.PetDocument) -> str:
        return self._input_formatter.output(document)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=self.steps)
        assert len(document.relations) == 0
        return self._output_formatter.parse(document, string)
