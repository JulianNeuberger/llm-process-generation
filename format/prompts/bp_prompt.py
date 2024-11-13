base_prompt = """
You are a business process modelling expert and you are tasked with creating a business process. You are creating one business process with a headline, that summarizes the business process. Don't use any bulletpoints, formulate whole sentences please. The BP (business process) should be a detailed description of a singular process. The business process should be from a real scenario. The field of the process is {topic}.
{and_prompt}
{examples_prompt}
{format_instructions}

"""

and_prompt = """ 
The BP can also have a parrallel workstream integrated or a decision that gives multiple options to continue the process (use phrases like "while", "meanwhile", "at the same time" or similiar and not only the word parrallel).
"""

examples_prompt = """
Here is one example:
Evaluation of customer satisfaction
Customer satisfaction is the primary criterion for the appraisal of the results of the CCH management system. The system for obtaining a quick and direct feedback from customers is based on the customer-orientated sales organization.
This organization includes the Customer Centre, where transactions are processed. A project coordinator and a member of our outside sales staff, who are assigned directly to the customer, accompany each transaction throughout, i.e. until delivery.
Customer training courses, seminars and trade fairs.
Regular customer training courses and contacts at trade fair presentations are also used to obtain information to promote customer satisfaction. Complaint management system
The EDP-supported complaint data bank is called upon as an objective parameter and as a source of information for assessing customer satisfaction.

"""

topics_prompt = """
You are a business process modelling expert and you are tasked with giving back different business fields, where we can find different business processes. Give me back a List with {count} different fields. 
{format_instructions}


"""