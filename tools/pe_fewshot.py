from langchain.chains import GraphCypherQAChain
# tag::import-prompt-template[]
from langchain.prompts.prompt import PromptTemplate
# end::import-prompt-template[]

from llm import llm
from graph import graph

#ON CREATE SET policyClass.`type`="policy_class"
cypher_create_pe_template = """
Task: Generate Cypher statement to create node in a graph database.
Instructions:
Use only the provided policy element in the schema.
Do not use any other other entity not provided as policy element.
Schema:
{schema}
Cypher examples:
1. The Principal System Administrator (PSA) creates  the policy classes Legal PC.
```
MERGE (pc:PolicyClass {{name: Legal PC}})
RETURN pc.name
```

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""


cypher_prompt = PromptTemplate.from_template(cypher_create_pe_template)


cypher_pe = GraphCypherQAChain.from_llm(
    llm,          # <1>
    graph=graph,  # <2>
    verbose=True,
    cypher_prompt=cypher_prompt
)
