quishpi_re_prompt = """As a business process modelling expert, your task involves analyzing textual descriptions of 
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