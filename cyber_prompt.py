import os
import openai
import openai
from openai import OpenAI

from src.cyber_prompt import *
from src.cyber_function import *

import tiktoken
from itertools import product
from dotenv import load_dotenv, find_dotenv


def ner_promt(input):
    system_message = """ \
Define: operations to create an entities or relations, entitie are categorized into five types: policy class (pc), \
user attribute (ua), object attribute (oa), users (u), and object (o). The are two major categories of relations: \
Assign or assignment and Association relations. The Assign relation has five types: user attribute (ua) assign to policy class (pc), \
object attribute (oa) assign to policy class (pc), user attribute (ua) assign to user attribute (ua), object attributes \
(oa) assign to object attribute (oa), user (u) assign to user attribute (ua), and object (o) assign to objects \
attribute (oa). The association relations is a triple of user attribute (ua), access right (ar), and object attribute (oa), \
where the access right is the action such as read, write, excute, create, and delete that the user attribute is allowed \
to perform on the object attribute (oa). \
If an operation is to create any of the five entity types, the response is structured as follows: \
'entity name | entity type | Explanation', where you explain the rationale behind the \
classification of the operation type. \
If an operation is to create any of five assignment relation, the response is structured as follows: \
'source entity name | source entity type | destination entity name | destination entity type | Explanation', where \
you identify the operation type and explain the source entity as the entity an assign relation is coming from and \
the destination entity is the entity assign relation is going to. \
If an operation is to create an association relation, the response is structured as follows: \
'user attribute name | access rights | object attribute name | Explanation', where you identify the operation type and \
explain the user attribute as the entity granted the access rights on an object attribute. \
The access rights is a subset of read (r), write (w), create and delete permissions on an object attribute. \
Question: Given the context below, identify the operation type and list the entity or entities for the operation."""



user_message1 = """ \
Context: "The Principal System Administrator (PSA) creates  the policy classes Legal PC and Finance PC in the database."
Given the context, identified the operation type and all required entities are:"""

assistant_message1 = """Answer:\
Legal PC | policy class | The operation is create entity, the sentence mentioned creating policy classes and Legal PC \
is one of the policy classes. \
Finance PC | policy class | The operation is create entity, This is another policy class in the create entity operation."""


user_message2 = """ \
Context: "The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept) \
user attribute to the Policy Class Finance PC" \
Given the context, identified the operation type and all required entities are:"""

assistant_message2 = """Answer:\
Finance Department | user attribute | Finance PC | policy class | This is a create an assignment operation, \
the statement specified creating an assignment relation between two entities. \
The Finance Department, user attribute is assigned to the Finance PC, a policy class. The Finance Department \
is a user attribute as it refers to a group within the organization characterized by their role in financial operations, \
thus fitting the user attribute category. \
Finance PC is a policy class as the name includes pc and is the only entity that can have assignment relations \
from both a user attribute and object attribute."""

user_message3 = """ \
Context: "An operation to assign Legal Records (Legal Rec) to the Legal Policy Class (Legal PC) is authorized \
when the requestor has an atribute of Principal System Administrator (PSA)."
Given the context, identified the operation type and all required entities are:"""

assistant_message3 = """Answer:\
Legal Records | object attribute | Legal Policy Class | policy class | This expression creates an assignment relation \
from an object attribute to a policy class. Legal Records is an object attribute as it is a resource for users and \
users' access to it is controlled. Legal policy class is a policy class as the name includes policy class \
and is the only entity that can have assignment relations from both a user attribute and object attribute.
"""

user_message4 = """ \
Context: "Authority to assign the users Bob and Jane to the user attributes Legal Counsel and Legal Advisor \
respectively is granted for the user that has an attribute of the Principal System Administrator (PSA)."
Given the context, identified the operation type and all required entities are:"""

assistant_message4 = """Answer:
PSA | user_attribute | True | As it specifies a role with the authority to assign user attributes, indicating a high-level administrative function within the system.
Legal Counsel | user_attribute | True | As it identifies a specific role or function that users like Bob can be assigned to, denoting a professional position within the legal domain.
Legal Advisor | user_attribute | True | As it denotes a specialized role within the legal field to which users such as Jane can be assigned, highlighting its significance as a user attribute.
Bob | U | True | As it is mentioned as an individual user who can be assigned to a specific role, thus recognized as a distinct entity within the user category.
Jane | U | True | As it is identified as an individual user eligible for assignment to a particular role, thereby qualifying as a unique entity in the user category.
"""



    user_test = f""" \
Context: {input}
Given the context, all relevant entities are:"""

    # delimiter = "####" # one token
    # Few-shot prompting
    messages=[
        {"role": "system",'content': system_message},
        # {"role": "user", "content": user_message1},
        # {"role": "assistant", "content": assistant_message1},
        # {"role": "user", "content": user_message2},
        # {"role": "assistant", "content": assistant_message2},
        {"role": "user", "content": user_message3},
        {"role": "assistant", "content": assistant_message3},
        {"role": "user", "content": user_message4},
        {"role": "assistant", "content": assistant_message4},
        {"role": "user", "content": user_test},
    ]

    return messages


def re_prompt(input_context, user_entity, object_entity):

    system_message = """ \
Please solve the Relation Extraction task. Given the context, consider what's the most precise relation between two entities belongs to the following N possible relations. \
The relation to choose must be in these N possible relations: 'r' for read, 'w' for write, 'x' for execute, 'c' for create, and 'd' for delete, 'assign' for assignment.
"""

    user_message1 = f""" \
Context: "A user ID can read and write all fields (SSN, Salary, Name, and Phone) in their own record."
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
Context: "as a Developer, I want to add the GTAS window data to the database, so that I can ensure the site is locked down during the GTAS submission period."
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

    user_message3 = f""" \
Context: "The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC"
Given the context, what are the relations between the user_attribute entity 'Fin Dept' and the PC entity 'Finance PC'?
1. user_attribute entity 'Fin Dept' is identified as a group or role within the organization that is involved in financial operations, representing a collection of users classified under a specific function related to finance.
2. PC entity 'Finance PC' is recognized as a set of access control policies or rules that govern the financial data and processes within the system, aimed at managing and protecting finance-related information.
3. According to the context, the phrase "The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC" clearly indicates an 'assign' relationship between the 'Fin Dept' user attribute and the 'Finance PC' policy class.
Therefore, the relation between user_attribute 'Fin Dept' and PC entity 'Finance PC' include 'assign' for assignment. \
"""
    assistant_message3 = """ \
Fin Dept | user_attribute | Finance PC | PC | assign. \
"""

    user_message4 = f""" \
Context: "The Principal System Administrator (PSA) request the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC"
Given the context, what are the relations between the U entity 'Bob' and the user_attribute entity 'Legal Counsel'?
1. U entity 'Bob' is identified as an individual user within the system, representing a person who can be assigned specific roles or attributes based on system policies or administrative decisions.
2. user_attribute entity 'Legal Counsel' is designated as a role or function within the organization, signifying a particular set of responsibilities or access rights related to legal affairs, and it serves as a classification for users who fulfill this role.
3. According to the context, the phrase "The Principal System Administrator (PSA) requests the assignment of the Finance Department (Fin Dept) user attribute to the Policy Class Finance PC" implies an administrative action where users or roles, such as 'Bob' being assigned to 'Legal Counsel', are systematically aligned with their respective functions or access rights, emphasizing the structured assignment process within the system's governance framework.
Therefore, the relation between U entity 'Bob' and user_attribute entity 'Legal Counsel' include 'assign' for assignment. \
"""
    assistant_message4 = """ \
Bob | U | Legal Counsel | user_attribute | assign.\
"""

    user_message5 = f""" \
Context: "The Principal System Administrator (PSA) can create an association relation that allows the Chief Finance Officer (CFO) to perform read and write on the Policies and Compliance object attribute."
Given the context, what are the relations between the user_attribute entity 'CFO' and the object_attribute entity 'Policies and Compliance'?
1. user_attribute entity 'CFO' is recognized as a high-ranking executive role within the organization, responsible for overseeing the financial operations and ensuring compliance with financial policies, which defines the CFO's authority and functional scope within the system.
2. object_attribute entity 'Policies and Compliance' is defined as a collection or category of data that outlines the regulatory and policy framework the organization must follow, embodying the rules, guidelines, and standards for financial and operational conduct.
3. According to the context, the phrase "The Principal System Administrator (PSA) can create an association relation that allows the Chief Finance Officer (CFO) to perform read and write on the Policies and Compliance object attribute" directly establishes that the CFO has been granted 'read' (r) and 'write' (w) permissions by the PSA on the 'Policies and Compliance' data, facilitating the management and updating of policy documents and compliance records as required by their role.
Therefore, the relation between user_attribute entity 'CFO' and object_attribute entity 'Policies and Compliance' include include 'r' for read and 'w' for write. \
"""
    assistant_message5 = """ \
CFO | user_attribute | Policies and Compliance | object_attribute | r.
CFO | user_attribute | Policies and Compliance | object_attribute | w. \
"""

    user_test = f""" \
Context: {input_context}
Given the context, what are the relations between the entity '{user_entity}' and the entity '{object_entity}'?"""

    # Few-shot prompting
    messages=[
            {"role": "system",'content': system_message},
            {"role": "user", "content": user_message1},
            {"role": "assistant", "content": assistant_message1},
            {"role": "user", "content": user_message2},
            {"role": "assistant", "content": assistant_message2},
            {"role": "user", "content": user_message3},
            {"role": "assistant", "content": assistant_message3},
            {"role": "user", "content": user_message4},
            {"role": "assistant", "content": assistant_message4},
            {"role": "user", "content": user_message5},
            {"role": "assistant", "content": assistant_message5},
            {"role": "user", "content": user_test},
    ]

    return messages
