VAN_DER_AA_PROMPT = """Your task is to extract declarative process models from natural language process descriptions. 
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

Please return raw text, do not use any formatting.
"""

# F1=0.81 auf 30 Beispielen mit 0 Shots extra
VAN_DER_AA_PROMPT_STEPWISE = """Your task is to extract declarative process models from natural language process descriptions. 
The process descriptions consist of a series of actions, each described by a predicate and an object. 
Your goal is to identify constraints between these actions, which dictate the ordering and existence of actions within
the process. Constraints can be one of the following types:

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

Additionally, you may encounter negations of constraints, indicated by statements like "do not" or "must not."

Please ensure the correct identification and formatting of constraints in the given text. Output one constraint 
per line, including whether the constraint is negated (TRUE or FALSE), the type of constraint, and the extracted 
head and tail actions separated by tabs. Stick closely to the provided examples and descriptions, and be careful 
to distinguish between precedence, response, and succession constraints.

Let's carry out your task STEPWISE:
- Step 1: First, extract actions contained in the input sentence and list them line by line (a description how you can 
identify actions is given in ACTIONS (Def.) above). 
- Step 2: Insert the following character sequence in the output: '#-#-#RESULT#-#-#'
- Step 3: Generate constraints based on the constraint types above (see CONSTRAINTS (Def.) for details). To do this, only use 
the actions that you defined before the mark ('#-#-#RESULT#-#-#'). The format MUST be as follows:
    Print one constraint per line, where you separate if the constraint is negative (TRUE if 
    the document describes a negation, else it reads FALSE), the type of the constraint, and the 
    extracted actions by tabs in the following form (<...> are placeholders): 
    <TRUE or FALSE>\t<constraint type>\t<head action>\t<tail action>.
    
** Important Restriction: ** 
Make sure that for Step 1 you generate actions in the form of 'predicate object', which usually consists
of a transitive verb and the corresponding object. It must be really some activity described in the same sentence.

Here are some examples for both input and expected output (separated by the following symbol: |):
    - Example 1: submit paper#-#-#RESULT#-#-#The process begins when the author submits the paper.|FALSE\tinit\tsubmit paper
    - Example 2: sign contract\nadvertise product#-#-#RESULT#-#-#After signing the contract, the product can be advertised.|FALSE\tprecedence\tsign contract\tadvertise product
    - Example 3: sign contract\nadvertise product#-#-#RESULT#-#-#After signing the contract, the product is advertised but never before.|FALSE\tsuccession\tsign contract\tadvertise product
    - Example 4: call manager\nforward request#-#-#RESULT#-#-#When the manager is called, the request needs to be forwarded to the secretary, too.|FALSE\tresponse\tcall manager\tforward request
    - Example 5: archive proposal#-#-#RESULT#-#-#The process is completed as soon as the proposal is archived|FALSE\tend\tarchive proposal
    - Example 6: nofify manager\nreject request#-#-#RESULT#-#-#After notifying the manager, the request must not be rejected.|TRUE\tresponse\tnofify manager\treject request 
    - Example 7: cover products#-#-#RESULT#-#-#If it rains, products are covered until they are needed.|FALSE\texistence\tcover products
    - Example 8: cover products#-#-#RESULT#-#-#If it is sunny, products are NOT covered but remain in the yard until they are needed.|FALSE\tabsence\tcover products
    - Example 9: archive offer\nsend offer#-#-#RESULT#-#-#If the offer is archived, it is not sent and if it is sent, archiving is unnecessary.|FALSE\tnoncooccurrence\tarchive offer\tsend offer

Please return raw text, do not use any formatting.
"""
