import random
import typing

import langchain_core.prompt_values
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

improvement_prompt_template = """
** CONTEXT: **
You are a prompt engineer for information extraction (entities and their relations) from natural language texts that 
describe business processes. There is already a draft for such a prompt but giving you this prompt does not produce 
satisfying results.  

* You were tasked with identifying said information by this PROMPT:*
```
{original_prompt}
```

* You were given, for instance, the following INPUTS (ids in square brackets): *
{inputs}

* These are the corresponding EXPECTED outputs (mapped by input ids in square brackets): *
{gold}

* You PREDICTED the following corresponding outputs (mapped by input ids in square brackets, contains both a few correct and many wrong predictions!): * 
{predictions}

** YOUR TASK NOW IS: **  
There were a lot of predictions, which were wrong. Please generate an improved prompt
and reflect on the reasons for the changes you then made.

"""

improvement_prompt_focus_extension_template = """
Especially focus on the following information types, since they are currently the driving factors for the rather low
performance:
{main_painpoints}
"""


def run_quick_prompt():
    res = model.invoke(quick_prompt)
    print(res.content.__str__())


def trial_run():
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


def compile_prompt_improvement_message(given_original_prompt: str, given_inputs: str, gold: str, predictions: str,
                                       main_painpoints: str = "") -> langchain_core.prompt_values.PromptValue:
    if main_painpoints:
        improvement_prompt = prompts.PromptTemplate(
            template=f'{improvement_prompt_template}\n{improvement_prompt_focus_extension_template}',
            input_variables=["original_prompt", "inputs", "gold", "predictions", "main_painpoints"]
        )
        return improvement_prompt.format_prompt(
            original_prompt=given_original_prompt,
            inputs=given_inputs,
            gold=gold,
            predictions=predictions,
            main_painpoints=main_painpoints)
    else:
        improvement_prompt = prompts.PromptTemplate(
            template=improvement_prompt_template,
            input_variables=["original_prompt", "inputs", "gold", "predictions"]
        )
        return improvement_prompt.format_prompt(
            original_prompt=given_original_prompt,
            inputs=given_inputs,
            gold=gold,
            predictions=predictions)


def compile_inputs(given_inputs: typing.List[str]) -> str:
    result = ''
    for num, one_input in enumerate(given_inputs):
        result = f'{result}[{num}]:\t{one_input}\n'
    return result


def compile_annotations(annotations: typing.List[typing.List[str]]):
    compiled_annotations = ''
    for idx, annotations_of_one_document in enumerate(annotations):
        annotations_of_one_document_as_string = "\n\t\t".join(annotations_of_one_document)
        compiled_annotations = f'{compiled_annotations}[{idx}]:\t{annotations_of_one_document_as_string}\n'
    return compiled_annotations


if __name__ == '__main__':
    def main():
        print('Running')
        original_prompt = """
        You are a business process modelling expert, tasked with identifying
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

        results = [
            {
                "input": "Whenever the sales department receives an order, a new process instance is created.",
                "gold": ["FALSE	existence	receive order"],
                "prediction": ["FALSE	response	receive order	create new process instance"]
            },
            {
                "input": "A member of the sales department can then reject or accept the order for a customized bike.",
                "gold": ["FALSE	noncooccurrence	reject order	accept order"],
                "prediction": ["FALSE	existence	reject the order",
                               "FALSE	existence	accept the order"]
            },
            {
                "input": "In the former case, the process instance is finished.",
                "gold": ["FALSE	end	finish process instance"],
                "prediction": ["FALSE	end	finish process instance"]
            },
            {
                "input": "In the latter case, the storehouse and the engineering department are informed.",
                "gold": ["FALSE	existence	inform"],
                "prediction": ["FALSE	existence	inform storehouse",
                               "FALSE	existence	inform engineering department"]
            },
            {
                "input": "The storehouse immediately processes the part list of the order and checks the required quantity of each part.",
                "gold": ["FALSE	existence	check required quantity of each part"],
                "prediction": ["FALSE	existence	process part list",
                               "FALSE	existence	check required quantity"]
            },
            {
                "input": "If the part is available in-house, it is reserved.",
                "gold": ["FALSE	precedence	the part is available in-house	reserve"],
                "prediction": ["FALSE	response	part is available in-house	reserve part"]
            },
            {
                "input": "This procedure is repeated for each item on the part list.",
                "gold": [],
                "prediction": ["FALSE	existence	repeat procedure"]
            }
            ,
            {
                "input": "In the meantime, the engineering department prepares everything for the assembling of the ordered bicycle.",
                "gold": ["FALSE	existence	prepare"],
                "prediction": ["FALSE	existence	prepares everything for the assembling"]
            }
            ,
            {
                "input": "Afterwards, the sales department ships the bicycle to the customer and finishes the process instance.",
                "gold": ["FALSE	precedence	ship bicycle	finish process"],
                "prediction": ["FALSE	precedence	ship bicycle	finish process instance",
                               "FALSE	end	finish process instance	"]
            }
            ,
            {
                "input": "An electronic service then determines the significance of the customer based on information that has been collected during the history of the contractual relationship.",
                "gold": ["FALSE	existence	determine the significance of the customer"],
                "prediction": ["FALSE	existence	determine significance"]
            }
            ,
            {
                "input": "If some files are missing, a search is initiated, otherwise the files can be physically tracked to the intended location.",
                "gold": [
                    "FALSE	noncooccurrence	some files are missing	otherwise",
                    "FALSE	precedence	some files are missing	initiate search",
                    "FALSE	precedence	some files are missing	track"
                ],
                "prediction": [
                    "FALSE	response	missing files	initiate search",
                    "FALSE	precedence	initiate search	physically track files to the intended location"
                ]
            },
            {
                "input": "Each morning, the files which have yet to be processed need to be checked, to make sure they are in order for the court hearing that day.",
                "gold": ["FALSE	existence	check files"],
                "prediction": [
                    "FALSE    init    check the files",
                    "FALSE    response    check the files    make sure they are in order",
                    "FALSE    existence    make sure they are in order"
                ]
            },
            {
                "input": "After that, some other MC internal staff receives the physical SCT file (out of scope).",
                "gold": ["FALSE	existence	receive physical SCT file"],
                "prediction": ["FALSE	existence	receive the physical SCT file"]
            },
            {
                "input": "Then, Back Office attaches the new SCT document, and stores the expanded SCT physical file.",
                "gold": ["FALSE	precedence	attach new SCT document	store SCT physical file"],
                "prediction": [
                    "FALSE	succession	attach the new SCT document	store the expanded SCT physical file"]
            },
            {
                "input": "When the report is received, the respective SCT file is retrieved.",
                "gold": ["FALSE	existence	retrieve respective SCT file"],
                "prediction": ["FALSE	precedence	receive report	retrieve SCT file"]
            },
            {
                "input": "The SCT physical file is stored by the Back Office awaiting a report to be sent by the Police.",
                "gold": ["FALSE	existence	store SCT physical file"],
                "prediction": [
                    "FALSE    existence    store SCT physical file",
                    "FALSE    response    store SCT physical file    send report by Police"
                ]
            },
            {
                "input": "Then, the SCT Warrant Possession is forwarded to Queensland Police.",
                "gold": ["FALSE	existence	forward SCT Warrant Possession"],
                "prediction": ["FALSE	existence	forward the SCT Warrant Possession"]
            },
            {
                "input": "The Client Service Back Office as part of the Small Claims Registry Operations receives the request and retrieves the SCT file.",
                "gold": ["FALSE	existence	retrieve SCT file"],
                "prediction": ["FALSE	succession	receive request	retrieve SCT file"]
            },
            {
                "input": "The party sends a warrant possession request asking a warrant to be released.",
                "gold": ["FALSE	existence	request warrant possession"],
                "prediction": ["FALSE	existence	send warrant possession request"]
            },
            {
                "input": "After the assessment, if the risk is high, an approval must be requested, but if the risk is low the credit is accepted.",
                "gold": [
                    "FALSE	succession    the risk is high    request approval",
                    "FALSE	noncooccurrence    the risk is high    the risk is low",
                    "FALSE	precedence    the risk is low    accept credit",
                ],
                "prediction": [
                    "FALSE	precedence    assessment    request approval",
                    "FALSE	precedence    assessment    accepted"
                ]
            },
            {
                "input": "After the customer then receives the report about service performance and problem resolution from Customer Service, the process flow at the customer also ends.",
                "gold": ["FALSE	precedence	receive report	end process flow"],
                "prediction": ["FALSE	precedence	receive report	ends process flow"]
            },
            {
                "input": "After all three activities are completed the process ends within Customer Service.",
                "gold": [],
                "prediction": ["FALSE	end	the process ends within Customer Service"]
            },
            {
                "input": "Then, two concurrent activities are triggered, i.e. i) a report is created for the customer which details the current service performance and the resolution of the problem, and ii) an SLA violation rebate is reported to Billing & Collections who will adjust the billing.",
                "gold": ["FALSE	precedence	create report	report SLA violation rebate"],
                "prediction": ["FALSE	succession	create report	adjust billing"]
            },
            {
                "input": "Customer Service either receives the actual service performance (if there was no problem) or the problem resolution report.",
                "gold": [],
                "prediction": [
                    "FALSE	existence    receive actual service performance",
                    "FALSE	existence    receive problem resolution report"
                ]
            },
            {
                "input": "The trouble-shooting report is received by Service Management and this information goes then into the creation of the problem resolution report just as described for ii).",
                "gold": [
                    "FALSE	precedence    receive trouble - shooting report    create problem resolution report"],
                "prediction": [
                    "FALSE	succession	receive trouble-shooting report	create problem resolution report"]
            },
            {
                "input": "This report is sent out to Service Management, then the process ends.",
                "gold": ["FALSE	response	send report	end process"],
                "prediction": [
                    "FALSE	init    send out report to Service Management",
                    "FALSE	end    process ends"
                ]
            },
            {
                "input": "For the case that minor corrective actions are required, Service Management will undertake corrective actions by themselves.",
                "gold": [
                    "FALSE	precedence	minor corrective actions are required	undertake corrective actions"],
                "prediction": [
                    "FALSE	response	undertake corrective actions	undertake corrective actions by Service Management"]
            },
            {
                "input": "For the case that no problem was detected at all, the actual service performance is sent back to the Customer Service.",
                "gold": ["FALSE	precedence	no problem was detected	send back actual service performance"],
                "prediction": ["FALSE	existence	send back actual service performance"]
            },
            {
                "input": "Three alternative process paths may be taken.",
                "gold": [],
                "prediction": ["FALSE	existence	Three alternative process paths may be taken."]
            },
            {
                "input": "Subsequently it has to be determined what counter measures should be taken depending on the information in the final status report.",
                "gold": ["FALSE	existence	determine what counter measures should be taken"],
                "prediction": ["FALSE	init	determine what counter measures should be taken"]
            },
            {
                "input": "Service Management then prepares the final status report based on the received information.",
                "gold": ["FALSE	existence	prepare final status report"],
                "prediction": ["FALSE	existence	prepare final status report"]
            },
            {
                "input": "Either trouble report or the normal execution notification will be included in a status report and sent back to Service Management.",
                "gold": [],
                "prediction": [
                    "FALSE	existence    include in a status report",
                    "FALSE	existence    send back"
                ]
            },
            {
                "input": "If a problem is detected this will be analyzed by Resource Provisioning and a trouble report is created.",
                "gold": [
                    "FALSE	precedence    a problem is detected    analyze problem",
                    "FALSE	precedence    a problem is detected    create trouble report",
                    "FALSE	precedence    analyze problem    create trouble report"
                ],
                "prediction": [
                    "FALSE	precedence    detect problem    analyze problem",
                    "FALSE	succession    analyze problem    create trouble report"
                ]
            },
            {
                "input": "Taking together the information (i.e. contract commitment data + prioritized actions) a detailed problem report is created.",
                "gold": ["FALSE	existence	create a detailed problem report"],
                "prediction": [
                    "FALSE	existence    take together information",
                    "FALSE	existence    create detailed problem report"
                ]
            }
        ]

        inputs_as_string = compile_inputs([result['input'] for result in results])
        gold_as_string = compile_annotations([result['gold'] for result in results])
        pred_as_string = compile_annotations([result['prediction'] for result in results])

        improvement_message = compile_prompt_improvement_message(
            given_original_prompt=original_prompt,
            given_inputs=inputs_as_string,
            gold=gold_as_string,
            predictions=pred_as_string
        )

        print(improvement_message.to_string())

        print(model.get_num_tokens(improvement_message.to_string()))

        improved_prompt_result = model.invoke(improvement_message)
        print(improved_prompt_result.content.__str__())


    main()
