import asyncio
import streamlit as st
from utils import write_message
from chat_model import *



# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the NGAC Chatbot!  How can I help you?"},
    ]
# end::session[]
#
if "ngac_graph" not in st.session_state:
    st.session_state.ngac_graph = {
        "PE": {
            "U": set(),
            "UA": set(),
            "O": set(),
            "OA": set(),
            "PC": set()
        },
        "ASSIGNMENT": set(),
        "ASSOCIATION": set(),
        "PROHIBITION": set()
    }

def update_graph(spec_dict):
    relation_name = spec_dict["rel_name"]
    st.session_state.ngac_graph[relation_name].add(spec_dict["relation"])


def display_graph():
    open_response = "{"
    close_response = "}"
    response = ""
    for k, v in st.session_state.ngac_graph.items():
        if k == "PE":
            for kpe, vpe in st.session_state.ngac_graph["PE"].items():
                if vpe:
                    response += "{}: {}, ".format(kpe, vpe)
                    #print(vpe)
        elif v:
            response += "{}: {}, ".format(k, v)

    response = open_response + response + close_response
    return response

# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        #write_message('assistant', asyncio.run(prompt_processor(message)))
        spec_dict = asyncio.run(prompt_processor(message))
        update_graph(spec_dict)
        graph_state = display_graph()
        write_message('assistant', graph_state)
# end::submit[]


# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)
    print(f"@bot prompt = {prompt}")

    # Generate a response
    handle_submit(prompt)
# end::chat[]
