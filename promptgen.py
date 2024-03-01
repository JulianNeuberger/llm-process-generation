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

** Task **

You were tasked with identifying the information relevant for the data perspective of a process in a given input text. Please
write a concise prompt, that would help you extract the expected information.

** Input Text **

As a basic principle , ACME AG receives invoices on paper or fax . These are received by the Secretariat in the central inbox and forwarded after a short visual inspection to an accounting employee . In `` ACME Financial Accounting `` , a software specially developed for the ACME AG , she identifies the charging suppliers and creates a new instance ( invoice ) . She then checks the invoice items and notes the corresponding cost center at the ACME AG and the related cost center managers for each position on a separate form ( `` docket `` ) . The docket and the copy of the invoice go to the internal mail together and are sent to the first cost center manager to the list . He reviews the content for accuracy after receiving the copy of the invoice . Should everything be in order , he notes his code one on the docket ( `` accurate position - AP `` ) and returns the copy of the invoice to the internal mail . From it , the copy of the invoice is passed on to the next cost center manager , based on the docket , or if all items are marked correct , sent back to accounting . Therefore , the copy of invoice and the docket gradually move through the hands of all cost center managers until all positions are marked as completely accurate . However , if inconsistencies exist , e.g . because the ordered product is not of the expected quantity or quality , the cost center manager rejects the AP with a note and explanatory statement on the docket , and the copy of the invoice is sent back to accounting directly . Based on the statements of the cost center managers , she will proceede with the clarification with the vendor , but , if necessary , she consults the cost center managers by telephone or e-mail again . When all inconsistencies are resolved , the copy of the invoice is sent to the cost center managers again , and the process continues . After all invoice items are AP , the accounting employee forwards the copy of the invoice to the commercial manager . He makes the commercial audit and issues the approval for payment . If the bill amount exceeds EUR 20 , the Board wants to check it again ( 4 - eyes-principle ) . The copy of the invoice including the docket moves back to the accounting employee in the appropriate signature file . Should there be a complaint during the commercial audit , it will be resolved by the accounting employee with the supplier . After the commercial audit is successfully completed , the accounting employee gives payment instructions and closes the instance in `` ACME financial accounting `` .

** Expected Extracted Information **

- the content
- the copy of the invoice
- the invoice items
- the charging suppliers
- the approval for payment
- the clarification with the vendor
- his code
- the commercial audit
- it
- These
- The docket and the copy of the invoice

** Prompt **
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
