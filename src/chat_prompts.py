import os
import openai
import openai
from openai import OpenAI

import tiktoken
from itertools import product
from dotenv import load_dotenv, find_dotenv

def assign_fewshot(input):
    system_message = """ \
    Define: An input is categorized as assignment_request if it specifies creating (i.e.,assigning) an unlabelled relation from
    one entity to another entity or deassigning (i.e., deleting, removing) an unlabelled relation of one enitity from another entity.
    Classify an input as True if the request_type is an assignment_request, otherwise, False for an unknown_request.
    An input that mentions association or prohibition is not an assignment_request.
    A single response is structured as follows: 'input | request_type | True/False | Explanation',
    where you explain the rational behind the classification.
    A single response is limited  strictly to the given structure defined for this input.
    NO additional enumeration, explanation, or information is required.
    Question: Given the context below, identify the request type and explain the reason for choosing
    an assignment_request or unknown_request.
    """

    user_message1 = """\
    Context: "create the policy class Finance PC in the Database."
    """

    assistant_message1 = """ Answers: \
    create the policy class Finance PC in the Database. | unknown_request | False |
    The information in this context is different from the assignment_request expressions.
    """

    user_message4 = """\
    Context: "Assign the user attribute Chief Financial Officer (CFO) to the Finance Department."
    """

    assistant_message4 = """ Answer: \
    Assign the user attribute Chief Financial Officer (CFO) to the Finance Department. | assignment_request |
    True | This is an assignment_request. The word "Assign",  and two entities Chief Financial Officer (CFO) and Finance
    Department in this context, it indicates the creation of assignment relation from the  Chief Financial Officer (CFO) to the Finance Department.
    """

    user_message5 = """ \
    Context: "create an assignment of the object attribute Ledgers and Processes to the Finance Record."
    """

    assistant_message5 = """ Answer: \
    create an assignment of the object attribute Ledgers and Processes to the Finance Record. | assignment_request |
    True | There are two enities "Ledgers and Processes" and the "Finance Record" in this context and no access rights
    is granted or revoked. It indicates creating an assignment relation from Ledgers and Processes to the Finance Record
    """

    user_message6 = """ \
    Context: "deassign Contracts and Legal Documents (CLD) from the Legal Research."
    """

    assistant_message6 = """ Answer:\
    deassign Contracts and Legal Documents (CLD) from the Legal Research. | assignment_request | True | The term
    "deassign" in this context means deleting or removing an entity Contracts and Legal Documents (CLD) from
    another entity Legal Research.
    """

    user_test = f""" \
    Context: {input}
    Given the context, all request types are:"""

    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message1},
        {"role": "assistant", "content": assistant_message1},
        {"role": "user", "content": user_message4},
        {"role": "assistant", "content": assistant_message4},
        {"role": "user", "content": user_message5},
        {"role": "assistant", "content": assistant_message5},
        {"role": "user", "content": user_message6},
        {"role": "assistant", "content": assistant_message6},
        {"role": "user", "content": user_test},
    ]

    return messages

def assoc_fewshot(input):
    system_message = """ \
    Define: An input is an association_request type if it specifies creating (i.e., granting, authorizing)
    or revoking (i.e., deleting, removing) a relation labelled with access_rights from a user_attribute to an object_attribute.
    Classify an input as True if the request_type is an association_request, otherwise, False for an unknown_request.
    A single response is structured as follows: 'input | request_type | True/False | Explanation',
    where you explain the rational behind the classification.
    A single response is limited  strictly to the given structure defined for this input.
    NO additional enumeration, explanation, or information is required.
    Question: Given the context below, identify the request type and explain the reason for classifying input as
    an association_request or unknown_request.
    """

    user_message4 = """\
    Context: "Assign the user attribute Chief Financial Officer (CFO) to the Finance Department."
    """

    assistant_message4 = """ Answer: \
    Assign the user attribute Chief Financial Officer (CFO) to the Finance Department. | unknown_request | False |
    The expression creates a relation but it is unlabelled.
    """

    user_message6 = """\
    Context: "Grant the Human Resources Officer (HRO) the access right to create-ua, delete-assign, read from, and write to object attribute Finance Record (Fin Rec)."
    """

    assistant_message6 = """ Answer: \
    Grant the Human Resources Officer (HRO) the access right to delete, read from, and write to object
    attribute Finance Record (Fin Rec). | association_request | True | There are access right, two entities Chief Financial Officer (CFO) \
    and Finance Record (Fin Rec) in this Context.
    """

    user_message7 = """ \
    Context: "delete an association relation that allows the Chief Finance \
    Officer (CFO) to perform read and write on the Policies and Compliance object attribute."
    """

    assistant_message7 = """ Answer: \
    delete an association relation that allows the Chief Finance Officer (CFO)
    to perform read and write on the Policies and Compliance object attribute. | association_request | True |
    The "delete an association" indicates that revoking or deleting an association relation that grant Chief Finance Officer (CFO)
    the read and write operations (i.e., access_rights) on the Policies and Compliance object attribute.
    """

    user_message8 = """ \
    Context: "create an association that grants the Bookkeeper authority
    to perform create, delete, read, and write operations on Ledgers and Processes."
    """

    assistant_message8 = """ Answer: \
    create an association that grants the Bookkeeper authority
    to perform create, delete, read, and write operations on Ledgers and Processes. | association_request | True |
    This context indicates granting or authorizing the Bookkeeper on Ledgers and Processes to perform
    create, delete, read, and write operations.
    """

    user_test = f""" \
    Context: {input}
    Given the context, all request types are:"""

    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message4},
        {"role": "assistant", "content": assistant_message4},
        {"role": "user", "content": user_message6},
        {"role": "assistant", "content": assistant_message6},
        {"role": "user", "content": user_message7},
        {"role": "assistant", "content": assistant_message7},
        {"role": "user", "content": user_message8},
        {"role": "assistant", "content": assistant_message8},
        {"role": "user", "content": user_test},
    ]

    return messages

def prohibit_fewshot(input):
    system_message = """ \
    Define: An input request is categorized as a prohibition_request if states denying access rights
    such as read, write create and delete (i.e., permissions) from an entity, user_attribute, to another
    entity, object_attribute. The prohibition_request is a relation labelled with the denied or revoked
    access rights from a user_attribute to the object attribute.
    Classify an input as True if the request_type is a prohibition_request, otherwise, False for an unknown_request.
    A single response is structured as follows: 'input | request_type | True/False | Explanation',
    where you explain the rational behind the classification.
    A single response is limited  strictly to the given structure defined for this input.
    NO additional enumeration, explanation, or information is required.
    Question: Given the context below, identify the request type and explain the reason for classifying input as
    an association_request or unknown_request.
    """

    user_message9 = """ \
    Context: "Undo the prohibition that denies the Chief Finance Officer (CFO) the permission to create-assign, delete-assign, read, and write\
    on the resource Ledger and Processes."
    """

    assistant_message9 = """ Answer: \
    Undo the prohibition that denies the Chief Finance Officer (CFO) the permission to create-assign, delete-assign, read, and write\
    on the resource Ledger and Processes. | prohibition_request | True | The prohibition in this context indicates that this\
    request type is a prohibition_request.
    """

    user_message10 = """ \
    Context: "Decline the read and write access rights on the Private Records from any Contract Staff."
    """

    assistant_message10 = """ Answer: \
    Decline the read and write access rights on the Private Records from any Contract Staff. | prohibition_request |
    True | An input that request the declining or denying an action such as read, write, create and delete is a \
    prohibition_request type.
    """

    user_message11 = """ \
    Context: "Repeal the prohibition that disallows the Legal Advisor the access rights to create-assign, delete-assign, read, and write\
    on the directory Compliance and Regulation."
    """

    assistant_message11 = """ Answer: \
    Repeal the prohibition that disallows the Legal Advisor the access rights to create-assign, delete-assign, read, and write\
    on the directory Compliance and Regulation. | prohibition_request | True | When a prohibited access rights set is repealed or canceled
    then the request is a prohibition_request.
    """

    user_test = f""" \
    Context: {input}
    Given the context, all request types are:"""

    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message9},
        {"role": "assistant", "content": assistant_message9},
        {"role": "user", "content": user_message10},
        {"role": "assistant", "content": assistant_message10},
        {"role": "user", "content": user_message11},
        {"role": "assistant", "content": assistant_message11},
        {"role": "user", "content": user_test},
    ]

    return messages


def assignment_prompt(input):
    system_message = """ \
    Define: create and delete are operation types on the five categories of child to parent entity ordered pair of relations \
    given as: user to user_attribute (user, user_attribute), user_attribute \
    to user_attribute (user_attribute, user_attribute), user_attribute to policy_class (user_attribute, policy_class), \
    object to object_attribute (object, object_attribute), object_attribute to object_attribute (object_attribute, object_attribute), \
    and object_attribute to policy_class (object_attribute, policy_class). These operation types have no permissions, authority, \
    or association relations granting or revoking access rights.
    user is a given name of an individual and is a child entity to only a user_attribute parent entity \
    object is a resource user requests permission to access and is a child entity to only an object_attribute parent entity\
    user_attribute is a role or characteristic of a user or group of users is a child entity to another user_attribute \
    or a policy_class parent entity. \
    object_attribute is a name representing a collection object data or resources and is a child entity to another object_attribute  \
    or a policy_class parent entity. \
    policy_class is a root entity that has no parent and only a user_attribute or an object_attribute is a child entity to the policy_class.
    A single response is structured as follows: 'operation type | child entity name | child entity type | parent entity name | parent entity type \
    | Explanation', where you explain rationale behind the classification. A single response is limited
    strictly to the given structure defined for this input. No additional enumeration, explanation, or information is required beyond the given structure.
    """

    user_message1 = """ \
    Context: "Deassign the John from the Bookkeeper."
    Given the context, the operation type, child entity, parent entity are:
    """

    assistant_message1 = """ Answer: \
    delete | John | user | Bookkeeper | user_attribute | In this context, deassign is the same as delete \
    a pair of child to parent entity relation.
    John is a child entity name and it has a child entity type of a user because John is a person's (user's) name. \
    Bookkeeper is a parent entity name. It has parent type of user_attribute \
    since it describes the role of the user John and the only parent entity type for a child entity type of a user is \
    user_attribute
    """

    user_message2 = """ \
    Context: "Assign the ggc-movie file to genre."
    Given the context, the operation type, child entity, and parent entity are:
    """

    assistant_message2 = """ Answer: \
    create | ggc-movie | object | genre | object_attribute | Assign means create a child to parent entity \
    relation in this context.
    ggc-movie is a child entity name. The name ggc-movie is data file a user can access, hence,
    it has a child entity type of an object.
    what makes genre a parent entity name is that it categorizes ggc-movie and genre has an object_attribute \
    parent entity type since an object ggc-movie can only have an object_attribute parent entity.
    """

    user_message3 = """ \
    Context: "create the assignment of Bookkeeper to the Finance Department."
    Given the context, the operation type, child entity, and parent entity are:
    """

    assistant_message3 = """ Answer: \
    create | Bookkeeper | user_attribute | Finance Department | user_attribute | The operation type is \
    create as expressed in this context.
    Bookkeeper is a role of a user and it's an entity that belongs to the Finance Department. \
    Therefore Bookkeeper is a child entity name for a child entity type of user_attribute.
    Finance Department is a user_attribute that contains other user_attribute such as Bookkeeper. \
    The Finance Department is a parent entity name for a parent entity type of a user_attribute
    """

    user_test = f""" \
    Context: {input}
    Given the context, all operation type, child entity and parent entity are:"""


    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message1},
        {"role": "assistant", "content": assistant_message1},
        {"role": "user", "content": user_message2},
        {"role": "assistant", "content": assistant_message2},
        {"role": "user", "content": user_message3},
        {"role": "assistant", "content": assistant_message3},
        {"role": "user", "content": user_test},
    ]

    return messages

def association_prompt(input):
    system_message = """ \
    Define: create and delete are operation types on the ordered triple of user_attribute, access_rights, and object_attribute, (user_attribute, access_rights, object_attribute) called association relation.
    A create or delete operation on an association relation will create or delete a relation
    that grants or allows the user_attribute to perform the access_rights on the object_attribute.
    user_attribute is a role or characteristic of a user or group of users allowed to perform the set of actions in the access_rights of the triple.
    access_rights is a set of actions authorized for the user_attribute on the object_attribute of a given triple.
    These actions are read 'r', write 'w', create-assign 'c-assign', delete-assign 'd-assign'.
    object_attribute is a name representing a collection of object data or resources of the triple.
    creating an association relation, authorizing, or granting any access right indicates an operation type of a create an association relation.
    deleting association relation, revoking, or removing an allowed access right indicates an operation type of a delete an association relation.
    A single response is structured as follows: 'operation_type | user_attribute name | access_rights set | object_attribute name | Explanation',
    where you explain rationale behind the identified entities as an operation_type, user_attribute name, access_rights set, and object_attribute name.
    A single response is limited  strictly to the given structure defined for this input. NO additional enumeration, explanation, or information is required.
    """

    user_message_1 = """ \
    Context: "Grant the read, write, create-assign, and delete-assign on Ledgers to the Bookkeeper."
    Given the context, the operation_type, user_attribute name, access_rights set, and object_attribute name are:
    """

    assistant_message_1 = """Answer: \
    create | Bookkeeper | read, write, create-assign, delete-assign | Ledgers | In this context, grant implies to create operation_type. The entity granted the access_rights set is the Bookkeeper and the Bookkeeper is a role for the user. Therefore, the Bookkeeper is the user_attribute name. The access_rights set the are actions read, write, and create assign granted to the Bookkeeper this context. The object_attribute name is the Ledger. The Ledger is the entity on which the granted access is performed.
    """

    user_message_2 = """ \
    Context: "Revoke the permission for the Chief Accountant to read, write, create-assign, and delete-assign in the Finance Record."
    Given the context, the operation_type, user_attribute name, access_rights set, object_attribute name are:
    """

    assistant_message_2 = """Answer: \
    delete | Chief Accountant | read, write, create-assign, delete-assign | Finance Record | The operation_type is delete since the statement in this context referenced revoking permission.
    The Chief Accountant characterized the job function of an individual and it is the entity that permission is revoked, so, it is the user_attribute name.
    The access_rights set are read, write create-assign, and delete-assign since these are the actions revoked from the 'user_attribute name' Chief Accountant. A group of protected data or resources named Finance Record in this context is the object_attribute name.
    """

    user_message_3 = """ \
    Context: "Create an association from Legal Advisor to Compliance and Regulation with the create assign and delete assign access rights."
    Given the context, the operation_type, user_attribute name, access_rights set, and object_attribute name are:
    """

    assistant_message_3 = """Answer: \
    create | Legal Advisor | create-assign, delete-assign | Compliance and Regulation | The operation_type is create, as specified in this context. The association in this context refers to the triple, (user_attribute name, access_rights set, object_attribute name), it's a relation that starts from the user_attribute name to the object_attribute name and labelled with the access_rights set. Therefore, Legal Advisor is the user_attribute name, Compliance and Regulation is the object_attribute name, and  create-assign, delete-assign are the access_rights set.
    """

    user_message_4 = """ \
    Context: "revoke the Bookkeeper authority to perform create-assign, delete-assign, read, and write operations on Ledgers and Processes."
    Given the context, the operation_type, user_attribute name, access_rights set, and object_attribute name are:
    """

    assistant_message_4 = """Answer: \
    delete | Bookkeeper | create-assign, delete-assign, read, write | Ledgers and Processes | This context referenced revoke and it indicates
    the operation_type is a delete operation. The user_attribute name is the entity whose granted access is revoked, in this context,
    the Bookkeeper is the user_attribute name. Actions revoked from the user_attribute name Bookkeeper are
    create-assign, delete-assign, read, and write as mentioned in this context. The object_attribute name describes the
    collection of resources or group of data on which the allowed access is revoked. In this context, the object_attribute name is Ledgers and Processes
    """


    user_test = f""" \
    Context: {input}
    Given the context, all operation type, child entity and parent entity are:"""


    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message_1},
        {"role": "assistant", "content": assistant_message_1},
        {"role": "user", "content": user_message_2},
        {"role": "assistant", "content": assistant_message_2},
        {"role": "user", "content": user_message_3},
        {"role": "assistant", "content": assistant_message_3},
        {"role": "user", "content": user_message_4},
        {"role": "assistant", "content": assistant_message_4},
        {"role": "user", "content": user_test},
    ]

    return messages

def prohibition_prompt(input):
    system_message = """ \
    Define: create and delete are operation types on the ordered triple of user_attribute, access_rights, and object_attribute, (user_attribute, access_rights, object_attribute) called prohibition relation.
    prohibition relation is the opposite of association relation. A create or delete operation on a prohibition relation will create or delete a relation
    that denies or prohibits the user_attribute from performing the access_rights on the object_attribute.
    user_attribute is a role or characteristic of a user or group of users prohibited from performing the set of actions in the access_rights of the triple.
    access_rights is a set of actions disallowed for the user_attribute on the object_attribute of a given triple.
    These actions are read 'r', write 'w', create-assign 'c-assign', delete-assign 'd-assign'.
    object_attribute is a name representing a collection of object data or resources of the triple.
    creating prohibition relation, disallowing, or denying any access right indicates that an input is a create prohibition relation operation type.
    deleting prohibition relation, undoing, reversing, cancelling, or removing of any prohibited access right indicates that the input is a delete prohibition relation operation type.
    Structure a single response as follows: 'operation_type | user_attribute name | access_rights set | object_attribute name | Explanation', where you explain rationale behind
    the identified entities as an operation_type, user_attribute name, access_rights set, and object_attribute name.
    In the response structure, prefix the actions in the access_rights set with an exclamation mark (!).
    For example the read, write, create, delete actions in a access_rights set is structured as !read, !write, !create, !delete.
    A single response is limited  strictly to the given structure. NO additional enumeration, explanation, or information is required.
    """


    user_message_1 = """ \
    Context: "A doctor is prohibited from creating patients."
    Given the context, the operation_type, user_attribute name, access_rights set, and object_attribute name are:
    """

    assistant_message_1 = """ Answer: \
    create | doctor | !create | patients | In this context, the user_attribute is prohibited and it means that the
    operation type of the prohibition relation is a create. The context describes the entity disallowed to act on
    a resource as a doctor. It indicates that the user_attribute name is the doctor. '!create' is the access rights set,
    where the exclamation marks represents disallowing the action to create patients. Patients is the object attribute name of
    the triple, prohibition relaition in this context that performing a create action on is prohibited.
    """

    user_message_2 = """ \
    Context: "Undo the prohibition that denies the Chief Finance Officer (CFO) the permission to create-assign, delete-assign, read, and write\
    on the resource Ledger and Processes."
    """

    assistant_message_2 = """ Answer: \
    delete | Chief Finance Officer (CFO) | !create-assign, !delete-assign, !read, !write | Ledger and Processes |
    This context expressed the reversal of a prohibition relation through the word "undo" and that makes the
    operation type a delete. The Chief Finance Officer (CFO) is the user_attribute name that a set of restricted access Rights
    is been reversed. The create-assign, delete-assign, read, write are the actions in the access rights set and the
    exclamation marks that prefix the actions means that the Chief Finance Officer (CFO) is not allowed to perform these actions.
    The Ledger and Processes is the object_attribute name on which the restricted access is imposed.
    """

    user_message_3 = """ \
    Context: "Decline the read and write access rights on the Private Records from any Contract Staff."
    """

    assistant_message_3 = """ Answer: \
    create | Contract Staff | !read, !write | Private Records | This context expressed actions that needs to be declined
    and it means that the operation type is a create. The Contract Staff is the user_attribute name specified in the
    actions to prohibit. Disallowed read and write is represented as !read and !write and are the elements of the access rights set
    in this context. The data Private Records is the object_attribute name and it is the entity in this context that
    access rights set of actions are not allowed.
    """

    user_message_4 = """ \
    Context: "Repeal the prohibition that disallows the Legal Advisor the access rights to create-assign, delete-assign, read, and write\
    on the directory Compliance and Regulation."
    """

    assistant_message_4 = """ Answer: \
    delete | Legal Advisor | !create-assign, !delete-assign, r!ead, !write | Compliance and Regulation| This context expressed reversing a denied access rights
    and it means that the operation type is a delete. The Legal Advisor is the user_attribute name specified in the
    prohibited actions. Denied create-assign, delete-assign, read, and write is represented as !create-assign, !delete-assign, !read, !write and are the elements of the access rights set
    in this context. The resource Compliance and Regulation is the object_attribute name and it is the entity in this context that
    access rights set of actions are not allowed.
    """


    user_test = f""" \
    Context: {input}
    Given the context, all operation type, child entity and parent entity are:"""


    messages=[
        {"role": "system",'content': system_message},
        {"role": "user", "content": user_message_1},
        {"role": "assistant", "content": assistant_message_1},
        {"role": "user", "content": user_message_2},
        {"role": "assistant", "content": assistant_message_2},
        {"role": "user", "content": user_message_3},
        {"role": "assistant", "content": assistant_message_3},
        {"role": "user", "content": user_message_4},
        {"role": "assistant", "content": assistant_message_4},
        {"role": "user", "content": user_test},
    ]

    return messages
