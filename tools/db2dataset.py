import argparse
import json

import requests
import tiktoken

import os

from bs4 import BeautifulSoup
from deprecation import deprecated
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

API_SECRET_KEY = "xxx"
BASE_URL = "xxx"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

from langchain_community.document_loaders import PDFMinerLoader

from langchain.prompts import (
    PromptTemplate,
)
from langchain.agents import tool
os.environ["SERPER_API_KEY"] = "xxx"
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator
from typing import List
from typing import Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.utilities import GoogleSerperAPIWrapper

from langchain.agents import Tool
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
import langchain
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.chains import LLMChain
from selenium.common import TimeoutException
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, TokenTextSplitter

# import langchain.chains.retrieval_qa.base
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ["OPENAI_API_KEY"] = 'xxx'

os.environ["OPENAI_API_BASE"] = "xxx"
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from retry import retry
import re
import os
langchain.debug = False
import arxiv
import os
import ssl

import logging
import time
import urllib

from urllib.error import URLError
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from arxiv import SortCriterion, SortOrder
import os
import re
from langchain_community.llms import Ollama
import glob
from concurrent.futures import ProcessPoolExecutor
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from concurrent.futures import ThreadPoolExecutor
from collections import Counter


from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS

from tqdm import tqdm
from database.DBEntity import *
from retriever.arxiv_paper import Arxiv_paper
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker,declarative_base
import logging

import datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm


def build_dataset():
    engine = create_engine('xxxScImpactPredict')
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()

    results = session.query(PaperMapping).filter(PaperMapping.TNCSI != None, PaperMapping.TNCSI != -1).all()
    data = []

    for result in tqdm(results):
        paper_detail = session.query(PaperMapping_detail).filter(
            PaperMapping_detail.idLiterature_detail == result.idLiterature).first()
        if paper_detail:
            data.append({
                "title": result.title,
                "TNCSI": result.TNCSI,
                "abstract": paper_detail.abstract
            })

     
    df = pd.DataFrame(data)
    df.to_csv('data_for_model.csv', index=False)
    session.close()

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm
import pandas as pd
import numpy as np

def build_dataset_balanced():
    engine = create_engine('xxxScImpactPredict')
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()

     
    bins = np.linspace(0, 1, num=10)   
    labels = range(len(bins) - 1)

     
    counts = []
    for i in labels:
        count = session.query(PaperMapping).filter(
            PaperMapping.TNCSI > bins[i],
            PaperMapping.TNCSI <= bins[i + 1]
        ).count()
        counts.append(count)
    min_count = min(counts)

     
    data = []
    for i in labels:
        results = session.query(PaperMapping).filter(
            PaperMapping.TNCSI > bins[i],
            PaperMapping.TNCSI <= bins[i + 1]
        ).limit(min_count).all()

        for result in tqdm(results):
            paper_detail = session.query(PaperMapping_detail).filter(
                PaperMapping_detail.idLiterature_detail == result.idLiterature).first()
            if paper_detail:
                data.append({
                    "title": result.title,
                    "TNCSI": result.TNCSI,
                    "abstract": paper_detail.abstract
                })
    print(len(data))
     
    df = pd.DataFrame(data)
    df.to_csv(r'~\Documents\GitHub\ScImpactPredict\tools\data_for_model_balanced_2.csv', index=False)
    session.close()

# build_dataset_balanced()

build_dataset_balanced()
