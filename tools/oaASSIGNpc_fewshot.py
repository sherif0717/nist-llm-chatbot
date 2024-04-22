from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from llm import llm
from graph import graph


cypher_create_oaASSIGNpc_template = """
You are an expert in developing the NIST NGAC authorization graph model in the Neo4j graph database. You are tasked with identifying
an object attribute and policy class from a given policy expression.
The object attribute and the policy class represents two different types of graph nodes.
In the context of NIST NGAC, an object attribute refers to a characteristic or property associated with
a resource or object in a system in a given policy expression. Examples of object attributes may includes
the sensitivity level of a document, the file format of digital asset, the location of a physical object,
or any other relevant characteristic the influence access control policy decisions.
The NIST NGAC policy class is a grouping of access control policies that share common characteristics or
objectives. It serves as a high-level abstraction to organize and manage access control policies within a
system. For example, a system might have policy classes for "Departments", "confidentiality", "category",
"roles", "locations", and many more.
In the NIST NGAC authorization graph modelling, you can create an "assignment" (that is a graph edge) relation
from the object attribute node  to the policy class node.


Instructions:
You are prompted with the NIST NGAC access control policy expression,
Use only the identified polcicy class, object attribute as graph nodes and the edge between
them as assignment relation type in the schema.
Do not use any other other entity not identified as policy class, object attribute, or an assignment relation.
The schema generates the Neo4j cypher commands to communicate with the database and create graph policy nodes and edges.
Generate only the five Neo4j cypher commands given the cypher example below.

Schema:
{schema}
access policy expresson example:
1. The Principal System Administrator (PSA) request the assignment of the Finance Records (Fin Rec)
object attribute to the Policy Class Finance PC.

corresponding cypher example:
```
MERGE (ua:ObjectAttribute {{name: "Finance Records"}})
MERGE (pc:PolicyClass {{name: "Finance PC"}})
MERGE (oa) - [a:assign] -> (pc)
WITH oa, a, pc
MATCH (oa) - [a] -> (pc) RETURN type(a)
```
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate.from_template(cypher_create_oaASSIGNpc_template)


cypher_oaASSIGNpc = GraphCypherQAChain.from_llm(
    llm,          # <1>
    graph=graph,  # <2>
    verbose=True,
    cypher_prompt=cypher_prompt
)
