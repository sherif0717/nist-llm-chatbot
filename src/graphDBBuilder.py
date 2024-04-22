from graph import driver



# assignment_dict = {
# "op_type": None,
# "childPEName": None,
# "childPEType": None,
# "parentPEName": None,
# "parentPEType": None
# }


# triple_dict = {
# "op_type": None,
# "rel_name": None,
# "userAttributeName": None,
# "accessRightsSet": None,
# "objectAttributeName": None
# }

def createTripleRelation(tx, user_attribute, ar_set, object_attribute):
    tx.run("MERGE(ua:UserAttribute {name: $user_attribute}) "
    "MERGE(oa:ObjectAttribute {name: $object_attribute}) "
    "WITH ua, oa "
    "CALL apoc.create.relationship(ua, $ar_set, {}, oa) YIELD rel "
    "REMOVE rel.noOp",
    user_attribute=user_attribute, object_attribute=object_attribute, ar_set=ar_set)

def createAssignRelation(tx,assign_dict):
    sourceName = assign_dict["childPEName"]
    sourceType = assign_dict['childPEType']

    destinationName = assign_dict["parentPEName"]
    destinationType = assign_dict['parentPEType']

    op_type = assign_dict['op_type']

    if sourceType == 'user_attribute' and destinationType == 'policy_class':
        tx.run("MERGE(ua:UserAttribute {name: $sourceName}) "
        "MERGE(pc:PolicyClass {name: $destinationName}) "
        "MERGE(ua)-[:assign]->(pc)",
        sourceName=sourceName, destinationName=destinationName)
    elif sourceType == 'object_attribute' and destinationType == 'policy_class':
        tx.run("MERGE(oa:ObjectAttribute {name: $sourceName}) "
        "MERGE(pc:PolicyClass {name: $destinationName}) "
        "MERGE(oa)-[:assign]->(pc)",
        sourceName=sourceName, destinationName=destinationName)
    elif sourceType == 'user_attribute' and destinationType == 'user_attribute':
        tx.run("MERGE(uai:UserAttribute {name: $sourceName}) "
        "MERGE(uaj:UserAttribute {name: $destinationName}) "
        "MERGE(uai)-[:assign]->(uaj)",
        sourceName=sourceName, destinationName=destinationName)
    elif sourceType == 'object_attribute' and destinationType == 'object_attribute':
        tx.run("MERGE(oai:ObjectAttribute {name: $sourceName}) "
        "MERGE(oaj:ObjectAttribute {name: $destinationName}) "
        "MERGE(oai)-[:assign]->(oaj)",
        sourceName=sourceName, destinationName=destinationName)
    elif sourceType == 'user' and destinationType == 'user_attribute':
        tx.run("MERGE(u:User {name: $sourceName}) "
        "MERGE(ua:UserAttribute {name: $destinationName}) "
        "MERGE(u)-[:assign]->(ua)",
        sourceName=sourceName, destinationName=destinationName)
    elif sourceType == 'object' and destinationType == 'object_attribute':
        tx.run("MERGE(o:Object {name: $sourceName}) "
        "MERGE(oa:ObjectAttribute {name: $destinationName}) "
        "MERGE(o)-[:assign]->(oa)",
        sourceName=sourceName, destinationName=destinationName)

def writeToDB(op_dict):
    with driver.session() as session:
        if op_dict['rel_name'] == "PE" and op_dict["op_type"] == 'create':
            session.execute_write(createPolicyElement, op_dict)
        elif op_dict['rel_name'] == "ASSIGNMENT" and 'create' in op_dict["op_type"]:
            session.execute_write(createAssignRelation, op_dict)
        elif op_dict['rel_name'] == "ASSOCIATION" or op_dict["rel_name"] == "PROHIBITION":
            if 'create' in op_dict["op_type"]:
                ua = op_dict['userAttributeName']
                ar = op_dict['accessRightsSet']
                oa = op_dict['objectAttributeName']
                session.execute_write(createTripleRelation, ua, ar, oa)
