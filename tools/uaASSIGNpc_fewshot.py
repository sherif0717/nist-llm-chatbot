from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from llm import llm
from graph import graph

#ON CREATE SET policyClass.`type`="policy_class"
cypher_create_uaASSIGNpc_template = """
Context: "The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC"
Given the context, what are the relations between the U entity 'Bob' and the user_attribute entity 'Legal Counsel'?
1. Policy class entity 'Finance PC' is a a container for policy elements and relationships that pertain to a specific policy; every policy element is contained by at least one policy class.
2. user attribute entity 'Finance Department' is a policy element designated as a role or function within the organization, signifying a particular set of responsibilities or characterized a group of users, or it serves as a classification for users who fulfill this role.
3. According to the context, the phrase "The Principal System Administrator (PSA) requests the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC" implies an administrative action where users attribute or roles, such as 'Finance Department' being assigned to 'Finance PC'.
Therefore, the relation between the user attribute entity 'Finance Department' and policy class entity 'Finance PC' include 'assign' for assignment relation.

Instructions:
Use only the provided policy element and relation type in the schema.
Do not use any other other entity not provided as policy element or relation.
The schema generate commands to communicate with the database.
Schema:
{schema}
Cypher examples:
1. The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept)
user attribute to the Policy Class Finance PC.
```
MERGE (ua:UserAttribute {{name: "Finance Department"}})
MERGE (pc:PolicyClass {{name: "Finance PC"}})
MERGE (ua) - [a:assign] -> (pc)
WITH ua, a, pc
MATCH (ua) - [a] -> (pc) RETURN type(a)
```
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""


cypher_prompt = PromptTemplate.from_template(cypher_create_uaASSIGNpc_template)


cypher_uaASSIGNpc = GraphCypherQAChain.from_llm(
    llm,          # <1>
    graph=graph,  # <2>
    verbose=True,
    cypher_prompt=cypher_prompt
)
"""
2. I want to create an assignment relation from Legal Department to Legal PC.
```
MERGE (ua:UserAttribute {{name: "Finance Department"}})
MERGE (pc:PolicyClass {{name: "Finance PC"}})
MERGE (ua) - [a:assign] -> (pc)
```
"""
