#import streamlit as st
import os
import streamlit as st
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph



graph = Neo4jGraph(
    url=st.secrets["NEO4JD_URI"],
    username=st.secrets["NEO4JD_USERNAME"],
    password=st.secrets["NEO4JD_PASSWORD"],
)


uri=st.secrets["NEO4JD_URI"]
username=st.secrets["NEO4JD_USERNAME"]
password=st.secrets["NEO4JD_PASSWORD"]
driver = GraphDatabase.driver(uri, auth=(username,password))
driver.verify_connectivity()
#with GraphDatabase.driver(uri, auth=auth) as driver:



'''
uri=st.secrets["NEO4J_URI"]
username=st.secrets["NEO4J_USERNAME"]
password=st.secrets["NEO4J_PASSWORD"]
auth=(username,password)

graph = GraphDatabase.driver(uri, auth=(username,password))
graph.verify_connectivity()
#with GraphDatabase.driver(uri, auth=auth) as driver:
url=os.environ["NEO4JD_URI"]
username=os.environ["NEO4JD_USERNAME"]
password=os.environ["NEO4JD_PASSWORD"]


graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)


'''
