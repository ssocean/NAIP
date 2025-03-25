import argparse
import json

import requests
import tiktoken

import os

from bs4 import BeautifulSoup
from deprecation import deprecated
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

API_SECRET_KEY = "xxxx"
BASE_URL = "x"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

from langchain_community.document_loaders import PDFMinerLoader

from langchain.prompts import (
    PromptTemplate,
)
from langchain.agents import tool
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


# loader = TextLoader(r'~\Desktop\PDF_Analysis\journal.txt', encoding='utf-8')
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
from sqlalchemy import create_engine, and_

from sqlalchemy.orm import sessionmaker,declarative_base
import logging

import datetime
def auto_make_directory(dir_pth: str):
    '''
    自动检查dir_pth是否存在，若存在，返回真，若不存在创建该路径，并返回假
    :param dir_pth: 路径
    :return: bool
    '''
    if os.path.exists(dir_pth):   
        return True
    else:
        os.makedirs(dir_pth)
        return False
def init_logger(out_pth: str = 'logs'):
    '''
    初始化日志类
    :param out_pth: 输出路径，默认为调用文件的同级目录logs
    :return: 日志类实例对象
    '''
     
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    auto_make_directory(out_pth)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s # %(message)s')
    handler.setFormatter(formatter)
     
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
     
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger = init_logger(r'r')
    logger.info("Start print log")  
    logger.debug("Do something")  
    logger.warning("Something maybe fail.") 
    logger.info("'key':'value'")
    '''
    return logger

logger = init_logger()
# @retry()
def retrieve_papers(query,target_date_former:datetime, target_date_latter:datetime,download_folder=None):
    if download_folder is not None:
        if not os.path.exists(download_folder) :
            os.makedirs(download_folder)
     
    engine = create_engine('xxxScImpactPredict')

    Base = declarative_base()
     
    # Base.metadata.create_all(engine)

     
    Session = sessionmaker(bind=engine)
    session = Session()
     
    # datetime_now_minus_one_day = datetime.datetime.now() - datetime.timedelta(days=1)

    # results = session.query(PaperMapping).all()
     
    search = arxiv.Search(
        query=query,
        max_results=float('inf'),
        sort_by=arxiv.SortCriterion.Relevance,   
        sort_order=arxiv.SortOrder.Descending,   
    )
    error_list = []
    for i, result in enumerate(search.results()):
        if '/' in result._get_default_filename():
            if '\\' in result._get_default_filename():
                continue
            continue
        # # if result.published.date() == datetime.datetime.now().date():
        # #     continue
        # if result.published.date() < target_date.date():#datetime_now_minus_one_day.date():
        #     break
        # else:#if result.published.date() == datetime.datetime.now().date():  # result.published.year>=2022:
        #     # print(result._get_default_filename())

        if result.published.replace(tzinfo=None) < target_date_latter and result.published.replace(tzinfo=None) > target_date_former:
            # print(result.title, result.published)
            arxiv_paper = Arxiv_paper(result, ref_type='entity')
            data = session.query(PaperMapping).filter(PaperMapping.arxiv_id == arxiv_paper.id).first()
            if data is None:  # current paper is not in the database
                doc = PaperMapping(arxiv_paper=arxiv_paper,search_by_keywords=query)
                doc.valid = -1
                doc.TNCSI = None
                session.add(doc)
                doc_detail = PaperMapping_detail(idLiterature=doc.idLiterature, arxiv_paper=arxiv_paper,
                                                 search_by_keywords=query)

                session.add(doc_detail)
                session.commit()
                # if download_folder:
                #     if not os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                #         logger.info(f'{arxiv_paper.id} is not existed in the disk, try downloading it. ')
                #         try:
                #             result.download_pdf(dirpath=download_folder)
                #             doc_detail = PaperMapping_detail(idLiterature=doc.idLiterature, arxiv_paper=arxiv_paper,
                #                                              search_by_keywords=query)
                #             doc.downloaded_pth = result._get_default_filename()
                #             session.add(doc)
                #             session.add(doc_detail)
                #             session.commit()
                #         except URLError as e:
                #             # print(e)
                #             if os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                #                 os.remove(os.path.join(download_folder, result._get_default_filename()))
                #             time.sleep(8)
                #             logger.warning(f'{result._get_default_filename()} download failed. Due to {e}')
                #         except KeyboardInterrupt:
                 
                #             logger.info("Operation interrupted by user.")
                #             if os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                #                 os.remove(os.path.join(download_folder, result._get_default_filename()))
                        # finally:
                        #     logger.info("Removing files due to errors in arxiv downloading")
                        #     if os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                        #         os.remove(os.path.join(download_folder, result._get_default_filename()))
                            # error_list.append(result._get_default_filename())
            else:

                data.search_by_keywords = query
                # # print('done')
                session.commit()
    session.close()

# retrieve_papers('all:cs.AI ANDNOT ti:"survey" ANDNOT ti:"review"',datetime.datetime(2021, 12, 31),datetime.datetime(2022, 12, 31) )

def ensemble_meta_info():
     
    engine = create_engine('xxxScImpactPredict')

    Base = declarative_base()
     
    Base.metadata.create_all(engine)

     
    Session = sessionmaker(bind=engine)
    session = Session()
    # results = session.query(PaperMapping).filter(PaperMapping.TNCSI == None and PaperMapping.gpt_keyword == None)#.all()#
    results = session.query(PaperMapping).filter(and_(PaperMapping.valid!=0,PaperMapping.TNCSI.is_(None), PaperMapping.gpt_keyword.is_(None)))
    for result in tqdm(tqdm(results)):
        s2paper = S2paper(result.title,filled_authors=False)
        if s2paper.citation_count is not None: # S2bug
            result.TNCSI = s2paper.TNCSI['TNCSI']
            result.gpt_keyword = s2paper.gpt_keyword

            # doc_detail = PaperMapping_detail(idLiterature=result.idLiterature, s2_paper=s2paper)
            paper_detail = session.query(PaperMapping_detail).filter(
                PaperMapping_detail.idLiterature_detail == result.idLiterature).first()

            paper_detail.s2_id = s2paper.s2id
            paper_detail.s2_tldr = s2paper.tldr
            paper_detail.s2_DOI = s2paper.DOI
            paper_detail.s2_pub_info = s2paper.publication_source
            paper_detail.s2_citation_count = s2paper.citation_count
            paper_detail.s2_reference_count = s2paper.reference_count
            paper_detail.s2_field = s2paper.field
            paper_detail.s2_influential_citation_count = s2paper.influential_citation_count
            paper_detail.authors_num = len(s2paper.authors)
            session.commit()
        else:
            result.valid = 0
            session.commit()
        # time.sleep(0.5)

ensemble_meta_info()