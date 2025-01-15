base_prompt = """
You are a business process modelling expert and you are tasked with creating a business process. You are creating one business process with a given headline, that summarizes the business process. Don't use any bullet points, formulate whole sentences please. The BP (business process) should be a detailed description of a singular process. The business process should be from a real scenario. The name of the process is {topic}.
{and_prompt}
{examples_prompt}
{format_instructions}

"""

and_prompt = """ 
The BP can also have a parallel workstream integrated(use phrases like "while", "meanwhile", "at the same time" and or similar) or a decision points that gives multiple options to continue the process."""

examples_prompt = """
Here are two examples:
Headline: Warrant Possession Request Handling and Police Coordination
Description: The party sends a warrant possession request asking a warrant to be released. The Client Service Back Office as part of the Small Claims Registry Operations receives the request and retrieves the SCT file. Then, the SCT Warrant Possession is forwarded to Queensland Police. The SCT physical file is stored by the Back Office awaiting a report to be sent by the Police. When the report is received, the respective SCT file is retrieved. Then, Back Office attaches the new SCT document, and stores the expanded SCT physical file. After that, some other MC internal staff receives the physical SCT file (out of scope).

Headline: Documentation and Justification Requirements:
Description: Change requests must document the provision that is proposed to be changed, the new language that is proposed and a justification as to the reason the change is to be made. The justification shall explain the problem being addressed, the advantage of the change, and any effect the change may have on existing equipment or other specifications or documents. The DCR may be returned to the originator at any step in the process if it is determined that inadequate information was provided for the DCR to be approved or rejected.

"""

topics_prompt = """
You are a business process modelling expert and you are tasked with giving back different business fields, where different business process can be founda . Give me back a List with {count} different fields. 
{format_instructions}
"""

bp_list_prompt = """
You are a business process modelling expert and you are tasked with creating headlines for processes.
Create a short list of real business processes headlines that you would typically find in a handbook or can be modelled in bpmn in the field of {topic} that summarize this process.
{format_instructions}
"""