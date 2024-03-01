QUISHPI_RE_PROMPT_HANDCRAFTED_TASK_SEPARATION = """
You are a business process modelling expert, tasked with identifying
constraints between actions in textual process descriptions. Processes consist of actions and, thus, textual process
descriptions are sentences that describe a short sequence of actions. Ordering and existence of actions depend on
constraints between them. Below you find further definitions for ACTIONS and CONSTRAINTS:

- ACTIONS: predicate and object describing a task. Predicate is usually a transitive verb, and object is 
          some physical or digital object on which is being acted on. 

- CONSTRAINTS: defines if and how actions can be executed. Always has a source / head 
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

Let's carry out your task STEPWISE:
- Step 1: First, identify actions contained in the input sentence (a description how you can identify actions is given in 
ACTIONS (Def.) above). 
- Step 2: Insert the following character sequence in the output: '#-#-#RESULT#-#-#'
- Step 3: Generate constraints based on the constraint types above (see CONSTRAINTS (Def.) for details). To do this, only use 
the actions that you defined before the mark ('#-#-#RESULT#-#-#'). The format MUST be as follows:
    Print one constraint per line, where you separate if the constraint is negative (TRUE if 
    the document describes a negation, else it reads FALSE), the type of the constraint, and the 
    extracted actions by tabs in the following form (<...> are placeholders): 
    <TRUE or FALSE>\t<constraint type>\t<head action>\t<tail action>.
    
** Important Restrictions: ** 
- Restriction 1: Make sure that for Step 1 you generate actions in the form of 'predicate object', which usually consists
of a transitive verb and the corresponding object. It must be really some activity described in the same sentence. 
- Restriction 2: Pay special attention to the existence constraint. Some sentences might to cover a precedence,
response, or succession constraint but if there is only one action in the sentence in such cases it is more likely an
existence constraint.
    
Please return raw text, do not use any formatting.
"""

# F1=0.49 auf 10 Beispielen mit 0 Shots extra
QUISHPI_RE_PROMPT_HANDCRAFTED_W_EXAMPLES_TUNED = """
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

Examples: Here are some examples for both input and expected output (separated by the following symbol: |):
    - Example 1: The process begins when the author submits the paper.|FALSE\tinit\tsubmit paper
    - Example 2: After signing the contract, the product can be advertised.|FALSE\tprecedence\tsign contract\tadvertise product
    - Example 3: After signing the contract, the product is advertised but never before.|FALSE\tsuccession\tsign contract\tadvertise product
    - Example 4: When the manager is called, the request needs to be forwarded to the secretary, too.|FALSE\tresponse\tcall manager\tforward request
    - Example 5: The process is completed as soon as the proposal is archived|FALSE\tend\tarchive proposal
    - Example 6: After notifying the manager, the request must not be rejected.|TRUE\tresponse\tnofify manager\treject request 
    - Example 7: If it rains, products are covered until they are needed.|FALSE\texistence\tcover products
    - Example 8: If it is sunny, products are NOT covered but remain in the yard until they are needed.|FALSE\tabsence\tcover products
    - Example 9: If the offer is archived, it is not sent and if it is sent, archiving is unnecessary.|FALSE\tnoncooccurrence\tarchive offer\tsend offer

* IMPORTANT: 
    - Remember: If you think that in a sentence an action A does not depend on any other actions, it is most likely
    an Existence constraint.

Please return raw text, do not use any formatting.
"""

# F1=0.49 auf 10 Beispielen mit 10 Shots extra | F1=0.41 auf 10 Beispielen und 0 Shots
QUISHPI_RE_PROMPT_HANDCRAFTED_W_EXAMPLES = """
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

Examples: Here are some examples for both input and expected output (separated by the following symbol: |):
    - Example 1: The process begins when the author submits the paper.|FALSE\tinit\tsubmit paper
    - Example 2: After signing the contract, the product can be advertised.|FALSE\tprecedence\tsign contract\tadvertise product
    - Example 3: After signing the contract, the product is advertised but never before.|FALSE\tsuccession\tsign contract\tadvertise product
    - Example 4: When the manager is called, the request needs to be forwarded to the secretary, too.|FALSE\tresponse\tcall manager\tforward request
    - Example 5: The process is completed as soon as the proposal is archived|FALSE\tend\tarchive proposal
    - Example 6: After notifying the manager, the request must not be rejected.|TRUE\tresponse\tnofify manager\treject request 
    - Example 7: If it rains, products are covered until they are needed.|FALSE\texistence\tcover products
    - Example 8: If it is sunny, products are NOT covered but remain in the yard until they are needed.|FALSE\tabsence\tcover products
    - Example 9: If the offer is archived, it is not sent and if it is sent, archiving is unnecessary.|FALSE\tnoncooccurrence\tarchive offer\tsend offer

Please return raw text, do not use any formatting.
"""

QUISHPI_RE_PROMPT_HANDCRAFTED_WO_EXAMPLES = """
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

# F1=0.42 auf 100 Beispielen
QUISHPI_RE_PROMPT_GEN_WO_EXAMPLES = """As a business process modelling expert, your task involves analyzing textual descriptions of 
business processes to identify key actions and the constraints that govern the order and execution of these actions. 
The goal is to clearly extract and categorize the actions and constraints to aid in the modeling of these processes. 
Please follow the guidelines below to ensure accurate and comprehensive extraction:

1. **Identifying Actions:**
   - An action consists of a predicate (usually a transitive verb) and an object (the entity being acted upon, which can 
   be physical or digital).
   - Look for explicit mentions of tasks or operations being performed as indicators of actions within the text.

2. **Understanding and Extracting Constraints:**
   - Constraints dictate the execution order or the existence of actions within a process. Each constraint has a type 
   and is defined by its relationship between actions (head action/source and tail action/target).
   - The CONSTRAINT TYPES are as follows:
     - **Init:** Marks the beginning of a process. Requires explicit mention of the process starting.
     - **End:** Indicates the end of a process. Requires explicit mention of the process ending.
     - **Precedence:** The tail action occurs only if the head action has already been executed.
     - **Response:** Execution of the head action necessitates the execution of the tail action.
     - **Succession:** The head and tail actions are executed in sequence, with the head action preceding the tail.
     - **Existence:** An action must occur at some point in the process without dependency on another action.
     - **Absence:** An action must not occur throughout the process, independent of other actions' execution.
     - **Noncooccurrence:** The head and tail actions cannot occur together in the process.

3. **Handling Negations:**
   - Pay special attention to negations which reverse the implied constraint (e.g., "if action A happens, action B does 
   NOT happen" indicates a negation of a normal Response constraint).
   - Mark these negations clearly as they significantly impact the process flow and constraint relationships.

4. **Handling Ambiguities:**
   - Response, Precedence, Succession, and Existence constraints can easily be mixed up. 
   - To differentiate between Response and Precedence, carefully analyze whether the tail action CAN be performed 
   because the head action has already been performed (Precedence), or whether the tail action MUST THEREFORE be 
   performed (Response). If BOTH is true, it is a Succession constraint.
   -  If you recognize an action in a sentence and there is no other action, it is NEVER a Precedence, Response or 
   Succession constraint. If you nevertheless identify a need to perform this activity, it is usually an Existence 
   constraint. 

5. **Format for Extraction:**
   - For each identified constraint, format your extraction as follows: 
     `<NEGATION (TRUE/FALSE)>	<CONSTRAINT TYPE (from list above)>	<HEAD ACTION>	<TAIL ACTION (if applicable)>`.
   - If a constraint does not involve a tail action (e.g., init, end, existence, absence), omit the tail action from the 
   format.

Please proceed with analyzing the given document according to these guidelines.
"""
