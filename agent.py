from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import GraphCypherQAChain
#from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from tools.pe_fewshot import cypher_pe
from tools.uaASSIGNpc_fewshot import cypher_uaASSIGNpc
from tools.oaASSIGNpc_fewshot import cypher_oaASSIGNpc
from langchain import hub
from graph import graph
from llm import llm



"""
cypher_create_pe_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_create_pe_template
)#input_variables=["schema", "question"],


chain_language_example = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True,
    cypher_prompt=cypher_create_pe_prompt
)


The Principal System Administrator (PSA) creates  the policy classes Legal PC


def cypherExample(prompt):
    chain_language_example.run(prompt)
"""


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Cypher PE",
        description="Create policy element pe as neo4j graph nodes using Cypher",
        func=cypher_pe,
        return_direct=False
    ),
    Tool.from_function(
        name="Cypher uaASSIGNpc",
        description="neo4j cypher command that creates an assignment relation between a user attribute and a policy class",
        func=cypher_uaASSIGNpc,
        return_direct=False
    ),
    Tool.from_function(
        name="Cypher oaASSIGNpc",
        description="from the policy expression identify object attribute and policy class as graph nodes, and \
        apply the neo4j cypher commands that creates an assignment relation between an object attribute and a policy class",
        func=cypher_uaASSIGNpc,
        return_direct=False
    )
]

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)


agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    )




def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})
    print(f"\n\nresponse = {response}\n\n")

    return response['output']
