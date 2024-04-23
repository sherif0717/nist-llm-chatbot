import os
import asyncio
import openai
from openai import AsyncOpenAI
from openai import OpenAI
import csv
import tiktoken
#from itertools import product
from dotenv import load_dotenv, find_dotenv
from openai_multi_client import OpenAIMultiClient
_ = load_dotenv(find_dotenv()) # read local .env file: save your openai api key in the local file named .env
openai.api_key  = os.environ['OPENAI_API_KEY']

from src.chat_prompts import *
#from src.graphDBBuilder import *


PROMPT = 0
TYPE = 1
PREDICT = 2

#openai client object for concurrent openai API calls
api = OpenAIMultiClient(endpoint="chat.completions", data_template={"model": "gpt-3.5-turbo"})

#openai client object for sequencial openai API calls
client = AsyncOpenAI(api_key=openai.api_key)


def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):

    #response = openai.ChatCompletion.create(
    api = OpenAI()
    response = api.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut
    )

    return response.choices[0].message.content.strip()



async def is_assignment_message(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):
    print(f"\n@is_assignment_message\n")

    #response = openai.ChatCompletion.create(
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, #
        )

    return response.choices[0].message.content.strip()



async def is_association_message(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):

    #response = openai.ChatCompletion.create(
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, #
    )

    return response.choices[0].message.content.strip()

async def is_prohibition_message(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):

    #response = openai.ChatCompletion.create(
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, #
    )

    return response.choices[0].message.content.strip()


def format_prompt_type(responses):
    for response in responses:
        prompt_dict = {}
        response_list = response.split("|")

        if "True" in response_list[PREDICT].strip(" "):
            prompt_dict["prompt"] = response_list[PROMPT].strip(" ")
            prompt_dict["type"] = response_list[TYPE].strip(" ")

            return prompt_dict
    return

def map_prompt_to_operation(prompt):
    spec_response = None
    if prompt["type"] == "entity_request":
        return policy_element_prompt(prompt)
    elif prompt["type"] == "assignment_request":
        assign_fewshot = assignment_prompt(prompt)
        for _ in range(1):
            spec_response = get_completion_from_messages(assign_fewshot, temperature=0.8)
    elif prompt["type"] == "association_request":
        assoc_fewshot = association_prompt(prompt)
        for _ in range(1):
            spec_response = get_completion_from_messages(assoc_fewshot, temperature=0.8)
    elif prompt["type"] == "prohibition_request":
        prohibit_fewshot = prohibition_prompt(prompt)
        for _ in range(1):
            spec_response = get_completion_from_messages(prohibit_fewshot, temperature=0.8)
    return spec_response


def refine_assignment_response(response):
    response_dict = {}
    res = None
    if "Answer" in response:
        response = response.split(":")[1].split("|")
    else:
        response = response.split("|")
    response_dict["op_type"] = response[0].strip(" ")
    response_dict["childPEName"] = response[1].strip(" ")
    response_dict["childPEType"] = response[2].strip(" ")
    response_dict["parentPEName"] = response[3].strip(" ")
    response_dict["parentPEType"] = response[4].strip(" ")
    response_dict["rel_name"] = "ASSIGNMENT"
    response_dict["relation"] = (response_dict["childPEName"], response_dict["parentPEName"])

    # if 'create' in response_dict["op_type"]:
    #     res = "The assignment relation ({}, {}) has been created".format(response_dict["childPEName"], response_dict["parentPEName"])
    # elif 'delete' in response_dict["op_type"]:
    #     res = "The assignment relation ({}, {}) has been deleted".format(response_dict["childPEName"], response_dict["parentPEName"])
    #writeToDB(response_dict)
    return response_dict

def refine_triple_response(response):
    response_dict = {}
    res = None
    if "Answer" in response:
        response = response.split(":")[1].split("|")
    else:
        response = response.split("|")
    response_dict["op_type"] = response[0].strip(" ")
    response_dict["userAttributeName"] = response[1].strip(" ")
    response_dict["accessRightsSet"] = "{}".format(response[2].strip(" "))
    response_dict["accessRightsSet"] = "{" + response_dict["accessRightsSet"] + "}"
    #ars = response_dict["accessRightsSet"]
    #print(f"accessRightsSet = {ars}")
    response_dict["objectAttributeName"] = response[3].strip(" ")
    response_dict["relation"] = (response_dict["userAttributeName"], response_dict["accessRightsSet"], response_dict["objectAttributeName"])

    if "!" in response_dict["accessRightsSet"]:
        response_dict["rel_name"] = "PROHIBITION"
    else:
        response_dict["rel_name"] = "ASSOCIATION"
    #
    # if "!" in response_dict["accessRightsSet"]:
    #     if 'create' in response_dict["op_type"]:
    #         res = "Disallowed access rights set {} is imposed on {} for {}".format(response_dict["accessRightsSet"], response_dict["objectAttributeName"], response_dict["userAttributeName"])
    #     elif 'delete' in response_dict["op_type"]:
    #         res = "The prohibited access rights {} on {} for {} is repealed".format(response_dict["accessRightsSet"], response_dict["objectAttributeName"], response_dict["userAttributeName"])
    #     response_dict["rel_name"] = "ASSOCIATION"
    # else:
    #     if 'create' in response_dict["op_type"]:
    #         res = "The access right(s) {} on {} is granted to {}".format(response_dict["accessRightsSet"], response_dict["objectAttributeName"], response_dict["userAttributeName"])
    #     elif 'delete' in response_dict["op_type"]:
    #         res = "The permission for {} to perform {} on {} has been revoked".format(response_dict["userAttributeName"], response_dict["accessRightsSet"], response_dict["objectAttributeName"])
    #     response_dict["rel_name"] = "PROHIBITION"
    #writeToDB(response_dict)
    return response_dict

async def handle_asyncio_exceptions(task):
    if not task.cancelled():
        try:
            ex_task = task.exception()
        except asyncio.cancelledError:
            print(f"{task} task was cancelled")
            return
    if not task.done():
        await task
    try:
        task_response_value = task.result()
    except Exception as e_task:
        print(f"{task} failed with: {e_task}")
        return

    ex_task = task.exception()
    print(f"{task} exception: {ex_task}")
    return


async def prompt_processor(message):
    """
    A coroutine that runs as the entry point for the asyncio program to translate Natural Language Access
    control Policy expressions to NIST NGAC specification using openai API.

    Argument:
    message -- a user prompt (NLACP expression) from the UI.

    Helper Functions:
    assign_fewshot, assoc_fewshot, and prohibit_fewshot -- imported functions from the src.chat_prompts module
    that provide in-context training to openai for classifying user's prompts as an assignment, association,
    and prohibition relations, respectively.

    is_assignment_message, is_association_message, and is_prohibition_message -- are task coroutines scheduled
    by the prompt_processor and await their completions from openai API calls.

    format_prompt_type -- Identifies the only positive (True) openai API responses from task coroutine calls
    and formats prompt to the response relation type.

    map_prompt_to_operation -- uses the openai API call to recognize the entities (operation type and policy elements)
    in the user prompt.

    refine_assignment_response, and refine_triple_response -- formats the identified policy elements in the user's
    prompt to NGAC specification.

    add_done_callback -- ensures termination of task coroutines.

    Return:
    spec_response -- a translation of user's prompt to NIST NGAC specification.
    """
    background_assign_response = set()
    background_assoc_response = set()
    background_prohibit_response = set()
    print(f"\n@prompt_processor meassage = {message}\n")
    assign_response = assoc_response = prohibit_response =  None
    spec_response = None
    assign_message = assign_fewshot(message)
    assoc_message = assoc_fewshot(message)
    prohibit_message = prohibit_fewshot(message)

    async with asyncio.TaskGroup() as tg:
        print("\n@TaskGroup\n")
        #assign_response = asyncio.run(is_assignment_message(assign_message, temperature=0.8))
        #assoc_response = asyncio.run(is_association_message(assoc_message, temperature=0.8))
        #prohibit_response = asyncio.run(is_prohibition_message(prohibit_message, temperature=0.8))

        assign_response = tg.create_task(is_assignment_message(assign_message, temperature=0.8))
        await assign_response
        background_assign_response.add(assign_response)
        handle_asyncio_exceptions(assign_response)

        assoc_response = tg.create_task(is_association_message(assoc_message, temperature=0.8))
        await assoc_response
        background_assoc_response.add(assoc_response)
        handle_asyncio_exceptions(assoc_response)

        prohibit_response = tg.create_task(is_association_message(assoc_message, temperature=0.8))
        await prohibit_response
        background_prohibit_response.add(prohibit_response)
        handle_asyncio_exceptions(prohibit_response)

    assign_response = assign_response.result()
    assoc_response = assoc_response.result()
    prohibit_response = prohibit_response.result()

    print(f"\n@chat_model assign_response = {assign_response}\n")
    print(f"\n@chat_model assoc_response = {assoc_response}\n")
    print(f"\n@chat_model prohibit_response = {prohibit_response}\n")

    responses = [assign_response, assoc_response, prohibit_response]

    formatted_prompt_type = format_prompt_type(responses)
    if formatted_prompt_type:
        spec_response = map_prompt_to_operation(formatted_prompt_type)
        print(f"\n@chat_model spec_response = {spec_response}\n")

        if "assignment_request" in assign_response:
            return refine_assignment_response(spec_response)
        elif "association_request" in assoc_response:
            return refine_triple_response(spec_response)
        elif "prohibition_request" in prohibit_response:
            return refine_triple_response(spec_response)

    assign_response.add_done_callback(background_assign_response.discard)
    assoc_response.add_done_callback(background_assoc_response.discard)
    prohibit_response.add_done_callback(background_prohibit_response.discard)

def main():
    pass
if __name__ == '__main__':
    main()
