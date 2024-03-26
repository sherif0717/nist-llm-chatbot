#import streamlit as st
import os
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from itertools import product
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


url=os.environ["NEO4JD_URI"]
username=os.environ["NEO4JD_USERNAME"]
password=os.environ["NEO4JD_PASSWORD"]

driver = GraphDatabase.driver(url, auth=(username,password))
'''
localGraph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
)
'''
