import random
import typing

import langchain_openai
from langchain_core import prompts

import data
import format

prompt_template = """
You are business process management expert, able to read textual descriptions
of business processes and identify information that is relevant for creating
formal business process models in Declare or BPMN. You are given pairs of 
textual business process descriptions (input) and annotations (output), please 
generate a prompt that would help a generative language model to create the 
output annotations given the input text.

{examples}"""

model_name = "gpt-4-0125-preview"
model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(model_name=model_name)

quick_prompt = """
You are business process management expert, able to read textual descriptions
of business processes and identify information that is relevant for creating
formal business process models in Declare or BPMN.

---

You were tasked with identifying information by this prompt:

** Prompt **

```
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

```

You were given this input:

** Input **

The Police Report related to the car accident is searched within the Police Report database and put in a file together with the Claim Documentation . This file serves as input to a claims handler who calculates an initial claim estimate . Then , the claims handler creates an Action Plan based on an Action Plan Checklist available in the Document Management system . Based on the Action Plan , a claims manager tries to negotiate a settlement on the claim estimate . The claimant is informed of the outcome , which ends the process .

You created these predictions:

** Predictions **

* ok *
(activity data, an Action Plan, [48, 49, 50])
(activity data, an initial claim estimate, [37, 38, 39, 40])
(actor, the claims handler, [44, 45, 46])
(actor, a claims manager, [70, 71, 72])
(activity, creates, [47])
(further specification, within the Police Report database, [10, 11, 12, 13, 14])
(activity data, The Police Report, [0, 1, 2])
(actor, The claimant, [83, 84])


* non ok *
(activity, which ends the process, [91, 92, 93, 94])
(activity, is searched, [8, 9])
(further specification, based on an Action Plan Checklist, [51, 52, 53, 54, 55, 56])
(further specification, serves as input, [28, 29, 30])
(further specification, on the claim estimate, [78, 79, 80, 81])
(further specification, available in the Document Management system, [57, 58, 59, 60, 61, 62])
(further specification, of the outcome, [87, 88, 89])
(xor gateway, Then, [42])
(actor, who calculates, [35, 36])
(activity data, an Action Plan, [53, 54, 55])
(activity, to negotiate a settlement, [74, 75, 76, 77])
(activity, tries, [73])
(activity, put in a file, [16, 17, 18, 19])
(actor, to a claims handler, [31, 32, 33, 34])
(activity data, This file, [26, 27])
(activity data, related to the car accident, [3, 4, 5, 6, 7])
(xor gateway, Based on the Action Plan, [64, 65, 66, 67, 68])
(and gateway, and, [15])
(further specification, together with the Claim Documentation, [20, 21, 22, 23, 24])
(activity, is informed, [85, 86])


* missing *
(actor, who, [35])
(activity, put, [16])
(activity, informed, [86])
(activity, searched, [9])
(activity data, a settlement, [76, 77])
(further specification, in a file together with the Claim Documentation, [17, 18, 19, 20, 21, 22, 23, 24])
(activity data, the outcome, [88, 89])
(activity, negotiate, [75])
(activity, calculates, [36])

There were a lot of predictions, which were wrong, how could i improve your prompt, so you are more accurate?

** Improved Prompt **
"""

res = model.invoke(quick_prompt)
print(res.content.__str__())
exit()

prompt = prompts.PromptTemplate(
    template=prompt_template, input_variables=["input", "output"]
)
importer = data.PetImporter("res/data/pet/all.new.jsonl")
documents = importer.do_import()
document = documents[0]

formatter = format.PetMentionListingFormattingStrategy(steps=["mentions"])

num_examples = 5
examples: typing.List[str] = []
for d in random.sample(documents, num_examples):
    examples.append(f"Input:\n{formatter.input(d)}\n\nOutput:\n{formatter.output(d)}")
formatted_examples = "\n\n".join(examples)

prompt_text = prompt.format(examples=formatted_examples)
res = model.invoke(prompt_text)
print(formatted_examples)
print()
print("===================================")
print()
print(res.content.__str__())
