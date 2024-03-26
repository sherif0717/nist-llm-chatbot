import os
import csv
import re
import pandas as pd
import openai
from openai import OpenAI

import tiktoken
from itertools import product
from dotenv import load_dotenv, find_dotenv
from graph import driver

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

ARGSEQ_PE_LENGTH = ARGSEQ_ASSIGN_LENGTH = 3
ARGSEQ_ASSOC_LENGTH = 4

pe_map = {
            'pc': {'columnName': "Policy Classes (PC)", 'name': 'Policy Class', 'id': 'pc'},
            'ua': {'columnName': "User Attributes (UA)", 'name': 'User Attribute', 'id': 'ua'},
            'oa': {'columnName': "Object Attributes (OA)", 'name': 'Object Attribute', 'id': 'oa'},
            'u': {'columnName': "Users (U)", 'name': 'User', 'id': 'u'},
            'o': {'columnName': "Objects (O)", 'name': 'Object', 'id': 'o'}
        }

assign_map = {
            'uapc': {'columnName': 'UAassignPC', 'sourceId': 'ua', 'destinationId': 'pc'},
            'oapc': {'columnName': 'OAassignPC', 'sourceId': 'oa', 'destinationId': 'pc'},
            'uaua': {'columnName': 'UAassignUA', 'sourceId': 'ua', 'destinationId': 'ua'},
            'oaoa': {'columnName': 'OAassignOA', 'sourceId': 'oa', 'destinationId': 'oa'},
            'uua': {'columnName': 'UassignUA', 'sourceId': 'u', 'destinationId': 'ua'},
            'ooa': {'columnName': 'OassignOA', 'sourceId': 'o', 'destinationId': 'oa'}
}


def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo", #model="gpt-4-0125-preview",
                                 temperature=0,
                                 max_tokens=500):

    #response = openai.ChatCompletion.create(
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut
    )

    '''
    token_dict = {
'prompt_tokens':response['usage']['prompt_tokens'],
'completion_tokens':response['usage']['completion_tokens'],
'total_tokens':response['usage']['total_tokens'],
    }
    '''

    #return response.choices[0].message["content"]
    #return response.choices[0].message.content#["content"]
    return response.choices[0].message.content.strip()

# Function to extract entity information from structured text and save it into a dictionary
def extract_entities_from_structured_text(text):
    # Split the text into lines as each line contains one entity information
    lines = text.strip().split('\n')

    # Filter lines that contain '|'
    processed_lines = [line for line in lines if '|' in line]
    # processed_lines = '\n'.join(processed_lines)
    # print(processed_lines)

    entities_dict = {}

    # Process each line to extract entity information
    for line in processed_lines:
        # Splitting the line by '|' to extract different parts of the entity information
        parts = [part.strip() for part in line.split('|')]

        # Extract entity name, type, is_entity flag, and explanation
        entity_name = parts[0]
        entity_type = parts[1]
        is_entity = parts[2] == "True"  # Convert string "True"/"False" to boolean True/False
        explanation = parts[3]

        # Store the entity information in the dictionary using the entity name as key
        entities_dict[entity_name] = {
            'type': entity_type,
            'is_entity': is_entity,
            'explanation': explanation
        }

    return entities_dict

def extract_re(text):
    lines = text.strip().split('\n')
    processed_lines = [line for line in lines if '|' in line]
    processed_lines = '\n'.join(processed_lines)

    return processed_lines


def ner_promt(input):
    system_message = """ \
Define: An entity is defined as one of two types: a user role (termed "user_attribute") or a data field (termed "object_attribute"). \
Each entity must be clearly classified as either "user_attribute" or "object_attribute". \
The response should be structured as follows: 'entity name | entity type | True/False | Explanation', where you explain the rationale behind the classification. \
Question: Given the context below, identify a list of possible entities and for each entry explain why it either is or is not an entity:"""

    user_message1 = """ \
Context: Employee users can read Name and Phone fields of all records in EmployeeTable.
Given the context, all relevant entities are:"""

    assistant_message1 = """Answer:
Employee | user_attribute | True | As it specifies users within the system granted read permissions.
Name | object_attribute | True | As it is identified as a field within EmployeeTable accessible by users.
Phone | object_attribute | True | As it is recognized as a field within EmployeeTable that users can access."""

    user_message2 = """ \
Context: As a Developer, I want to add the GTAS window data to the database, so that I can ensure the site is locked down during the GTAS submission period.
Given the context, all relevant entities are:"""

    assistant_message2 = """Answer:
Developer | user_attribute | True | As it denotes the role with permissions to modify the system, specifically for managing GTAS window data operations.
Database | object_attribute | True | As it represents the storage system being accessed and modified, containing the GTAS window data."""


    user_test = f""" \
Context: {input}
Given the context, all relevant entities are:"""

    # delimiter = "####" # one token
    # Few-shot prompting
    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message1},
        {"role": "assistant", "content": assistant_message1},
        {"role": "user", "content": user_message2},
        {"role": "assistant", "content": assistant_message2},
        {"role": "user", "content": user_test},]

    return messages


def re_prompt(input_context, user_entity, object_entity):

    system_message = """ \
Please solve the Relation Extraction task. Given the context, consider what's the most precise relation between two entities belongs to the following N possible relations. \
The relation to choose must be in these N possible relations: 'r' for read, 'w' for write, 'x' for execute, 'c' for create, and 'd' for delete.
"""

    user_message1 = f""" \
Context: A user ID can read and write all fields (SSN, Salary, Name, and Phone) in their own record.
Given the context, what are the relations between the user_attribute entity 'ID' and the object_attribute entity 'Name'?
1. user_attribute entity 'ID' is a unique identifier for a user, which refers to the entity of a system user in the context.
2. object_attribute entity 'Name' is a personal information field, which refers to the entity of stored personal names in user records in the context.
3. According to the context, The phrase 'A user ID can read and write all fields (SSN, Salary, Name, and Phone) in their own record' explicitly indicating that \
the user associated with a given ID has permissions to both read ('r') and write ('w') across all specified fields, including 'Name', within their personal record. \
Therefore, the relation between Subject entity 'Employee' and Object entity 'Name' include 'r' for read and 'w' for write. \
"""

    assistant_message1 = """ \
ID | user_attribute | Name | object_attribute | r.
ID | user_attribute | Name | object_attribute | w. \
"""

    user_message2 = f""" \
Context: s a Developer, I want to add the GTAS window data to the database, so that I can ensure the site is locked down during the GTAS submission period.
Given the context, what are the relations between the user_attribute entity 'Developer' and the object_attribute entity 'Database'?
1. user_attribute entity 'Developer' is a role responsible for software development and system updates, which refers to the entity of technical personnel in the context.
2. object_attribute entity 'Database' is a structured collection of data, which refers to the entity of a storage system in the context.
3. According to the context, the phrase "As a Developer, I want to add the GTAS window data to the database, so that I can ensure the site is locked down during the GTAS submission period" \
directly indicates the need for both 'read' (r) and 'write' (w) permissions for the Developer on the Database, through the actions of adding data and ensuring site lockdown.
Therefore, the relation between Subject entity 'Developer' and Object entity 'Database' include 'r' for read and 'w' for write. \
"""
    assistant_message2 = """ \
Developer | user_attribute | Database | object_attribute | r.
Developer | user_attribute | Database | object_attribute | w. \
"""

    user_test = f""" \
Context: {input_context}
Given the context, what are the relations between the user_attribute entity '{user_entity}' and the object_attribute entity '{object_entity}'?"""

    # Few-shot prompting
    messages=[
            {"role": "system",'content': system_message},
            {"role": "user", "content": user_message1},
            {"role": "assistant", "content": assistant_message1},
            {"role": "user", "content": user_message2},
            {"role": "assistant", "content": assistant_message2},
            {"role": "user", "content": user_test},
    ]

    return messages

def load_data(tx):
    tx.run("LOAD CSV WITH HEADERS FROM 'file:///netflix_titles.csv' AS line "
    "WITH line WHERE line.director IS NOT NULL "
    "MERGE(m:Movie {id: line.show_id,title: line.title, releaseYear: line.release_year})"
    "MERGE(p:Person {name: line.director})"
    "MERGE (p) -[:DIRECTED]-> (m)"
    )

def createPolicyElement(tx, pe_dict):
    for pe in pe_dict['argSeq']:
        pe_type = pe_dict['peType']
        pe_name = pe
        tx.run("MERGE(pc:PolicyClass {name: $pe_name})",
        pe_name=pe_name
        )

'''
"REMOVE source.noop "
"WITH source "
"CALL apoc.merge.node([$destinationType], {name:$destinationName}) YIELD destination "
"WITH destination "
"REMOVE destination.noop "
"CALL apoc.create.relationship(source, destination) YIELD rel "
"REMOVE rel.noOp",
'''

def createAssignRelation(tx,assign_dict):
    argSeq = assign_dict['argSeq']

    for i, pe in enumerate(argSeq):
        if i%2==0:
            sourceName = pe
            sourceType = assign_dict['sourceIdType']
            sourceId = assign_dict['sourceId']
            destinationName = argSeq[i+1]
            destinationType = assign_dict['destinationIdType']
            destinationId = assign_dict['destinationId']
            op_type = assign_dict['op_type']
            if sourceId == 'ua' and destinationId == 'pc':
                tx.run("MERGE(ua:UserAttribute {name: $sourceName}) "
                "MERGE(pc:PolicyClass {name: $destinationName}) "
                "MERGE(ua)-[:assign]->(pc)",
                sourceName=sourceName, destinationName=destinationName)
            elif sourceId == 'oa' and destinationId == 'pc':
                tx.run("MERGE(oa:ObjectAttribute {name: $sourceName}) "
                "MERGE(pc:PolicyClass {name: $destinationName}) "
                "MERGE(oa)-[:assign]->(pc)",
                sourceName=sourceName, destinationName=destinationName)
            elif sourceId == 'ua' and destinationId == 'ua':
                tx.run("MERGE(uai:UserAttribute {name: $sourceName}) "
                "MERGE(uaj:UserAttribute {name: $destinationName}) "
                "MERGE(uai)-[:assign]->(uaj)",
                sourceName=sourceName, destinationName=destinationName)
            elif sourceId == 'oa' and destinationId == 'oa':
                tx.run("MERGE(oai:ObjectAttribute {name: $sourceName}) "
                "MERGE(oaj:ObjectAttribute {name: $destinationName}) "
                "MERGE(oai)-[:assign]->(oaj)",
                sourceName=sourceName, destinationName=destinationName)
            elif sourceId == 'u' and destinationId == 'ua':
                tx.run("MERGE(u:User {name: $sourceName}) "
                "MERGE(ua:UserAttribute {name: $destinationName}) "
                "MERGE(u)-[:assign]->(ua)",
                sourceName=sourceName, destinationName=destinationName)
            elif sourceId == 'o' and destinationId == 'oa':
                tx.run("MERGE(o:Object {name: $sourceName}) "
                "MERGE(oa:ObjectAttribute {name: $destinationName}) "
                "MERGE(o)-[:assign]->(oa)",
                sourceName=sourceName, destinationName=destinationName)
        else:
            continue



#MERGE(ua)-[:ASSOC {access_right: $ar_set}]->(oa)
def createAssocRelation(tx, user_attribute, ar_set, object_attribute):
    tx.run("MERGE(ua:UserAttribute {name: $user_attribute}) "
    "MERGE(oa:ObjectAttribute {name: $object_attribute}) "
    "WITH ua, oa "
    "CALL apoc.create.relationship(ua, $ar_set, {}, oa) YIELD rel "
    "REMOVE rel.noOp",
    user_attribute=user_attribute, object_attribute=object_attribute, ar_set=ar_set)

def writeToDB(op_dict):
    with driver.session() as session:
        if op_dict['op_id'] in pe_map:
            session.execute_write(createPolicyElement, op_dict)
        elif op_dict['op_id'] in assign_map:
            session.execute_write(createAssignRelation, op_dict)
        elif op_dict['argSeq']:
            ua = op_dict['argSeq'][0]
            ar = op_dict['argSeq'][1]
            oa = op_dict['argSeq'][2]
            session.execute_write(createAssocRelation, ua, ar, oa)



'''
def addAssocRelations(assocDict):
    with driver.session() as session:
        print("ua = ", assocDict["ua"])
        print("ar = ", assocDict["ar"])
        print("oa = ", assocDict["oa"])
        ua = assocDict["ua"]
        ar = assocDict["ar"]
        oa = assocDict["oa"]
        session.execute_write(createAssocRelation, ua, ar, oa)
        #session.write_transaction(createAssocRelation, ua, ar, oa)
'''

def cleanAssocRelation(relation):
    relList = relation.split(",")
    assocList = []
    for word in relList:
        assocList.append(word.strip(" ").strip("{").strip("}").strip("(").strip(")"))
    assocLen = len(assocList)
    assocDict = {}
    assocDict["ua"] = ''.join(assocList[:1])
    assocDict["ar"] = "{" + ','.join(assocList[1:assocLen-1]) + "}"
    assocDict["oa"] = ''.join(assocList[-1])
    return assocDict

def getOperationAndArgSeq(request):
    """
    This function takes the argument request, i.e the access request column with their
    format - {(uai, {ars}, {arg_seq})}. The {ars} determines the length and type(s) of in their
    {argSeq}. Notations in {ars} = c/d-pe, c/d-assign-pe1pe2, c/d-assoc.

    Returns a dictionary. argSeq and op_id are keys with a values of a list and operation identifier
    """
    request_dict = {}
    if isinstance(request, str):
        r = request.split('{')
        r = [arg.strip(" ").strip("{").strip("}").strip("(").strip(")").strip("},").strip(",") for arg in r if str(arg)]
        request_dict['argSeq'] = r
        request_dict['op_id'] = r[1].split('-')[-1]
        #op_arg = r[1].split('-')
        #print(r)
        return request_dict #op_arg[-1]
    return None

def getArgSeq(request_dict):
    argSeqList = []
    op_id = None
    if request_dict['op_id']:
        op_id = request_dict['op_id']
    if op_id in pe_map:
        argSeqList = request_dict['argSeq'][ARGSEQ_PE_LENGTH-1].split(",")
        argSeqList = [arg.strip(' ') for arg in argSeqList]
    elif op_id in assign_map:
        argSeqList = request_dict['argSeq'][ARGSEQ_ASSIGN_LENGTH-1].split(",")
        argSeqList = [arg.strip(" ").strip("(").strip(")").strip(")}").strip("{(") for arg in argSeqList]
    else:
        argSeqList = request_dict['argSeq'][2:]
        argSeqList.append(argSeqList[1].split("}")[1].strip(", ").strip(")"))
        argSeqList[1] = "{"+argSeqList[1].split("}")[0]+"}"
    return argSeqList


def selectOperation(request_dict):
    op_id = None
    if request_dict:
        op_id = request_dict['op_id']
        request_dict['argSeq'] = getArgSeq(request_dict)
    else:
        return request_dict

    if op_id in pe_map:
        request_dict['op_type'] = 'c/d_pe'
        request_dict['peType'] = pe_map[op_id]['name']
    elif op_id in assign_map:
        request_dict['op_type'] = 'c/d_assign'

        request_dict['sourceId'] = assign_map[op_id]['sourceId']
        request_dict['sourceIdType'] = pe_map[request_dict['sourceId']]['name']

        request_dict['destinationId'] = assign_map[op_id]['destinationId']
        request_dict['destinationIdType'] = pe_map[request_dict['destinationId']]['name']
    else:
        request_dict['op_type'] = 'c/d_assoc'
    return request_dict


def readFile():
    df = pd.read_csv('FinOrgGraph.csv')
    df = df.reset_index()
    return df.iterrows()


def main():

    #test_string = "In addition to being able to read Name and Phone fields, HR users can read and write SSN and Salary fields of all records in EmployeeTable."
    #test_string = "As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles."
    df_object = readFile()
    for index, row in df_object:
        r_dict = getOperationAndArgSeq(row['Access Request'])
        #print(r_dict)

        op_dict = selectOperation(r_dict)
        #print(op_dict)



        if op_dict and 'argSeq' in op_dict:
            print(f"{op_dict}")
            writeToDB(op_dict)
        '''
        if op_dict['op_type'] == 'c/d_pe':
            op_dict['peNodeName'] = row[op_dict['columnName']]
            print("peNodeName = ", op_dict['peNodeName'])
        elif op_dict['op_type'] == 'c/d_assign':
            op_dict['SourceNodeName'] = row[op_dict['sourceColumnName']]
            op_dict['destinationIdName'] = row[op_dict['destinationColumnName']]
            print(f"SourceNodeName = {op_dict['SourceNodeName']}, destinationIdName = {op_dict['destinationIdName']}")
        '''
        #writeToDB(op_dict)
        #print(op_dict)
    '''
    with open('HospitalAccessControlPolicy.csv') as iFile:
        iCSV = csv.reader(iFile, delimiter='|')
        for row in iCSV:
            test_string = row[0]
            #test_string = "As a Data user, I want to have the 12-19-2017 deletions processed."
            print("INPUT:", test_string)
            tripleDict = cleanAssocRelation(','.join(row[1:]))
            if tripleDict["ua"] and tripleDict["ar"] and tripleDict["oa"]:
                addAssocRelations(tripleDict)
                print("Ground-truth:", tripleDict["ua"] + " " + tripleDict["ar"] + " " + tripleDict["oa"])
                print()
                #lineCount += 1

            # generate ner prompt
            ner_messages = ner_promt(test_string)

            for _ in range(5): # allows up to 5 attempts to successfully extract entities.
                response = get_completion_from_messages(ner_messages, temperature=0.8)
                try:
                    # Extract entities from the provided text data
                    extracted_entities = extract_entities_from_structured_text(response)
                    break
                except:
                    continue

            # Separate the entities into 'user_attribute' and 'object_attribute'
            user_attributes = [entity for entity, details in extracted_entities.items() if details['type'] == 'user_attribute']
            object_attributes = [entity for entity, details in extracted_entities.items() if details['type'] == 'object_attribute']

            # Generate all possible combinations of 'user_attribute' and 'object_attribute'
            combinations = list(product(user_attributes, object_attributes))
            print("Entities: ", combinations)
            print()

            for c in combinations:
                re_message = re_prompt(test_string,c[0],c[1])
                response = get_completion_from_messages(re_message, temperature=0.7)

                output = extract_re(response)
                print(output)
            print("____________________________")
            print()
    '''
if __name__ == '__main__':
    main()
