import json
import uuid
from datetime import datetime
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from retriever.Author import Author
from retriever.arxiv_paper import Arxiv_paper
from retriever.semantic_scholar_paper import S2paper


engine = create_engine('xxxScImpactPredict')
Base = declarative_base()



class PaperMapping(Base):
    __tablename__ = 'literature'   

    idLiterature = Column(String(36), primary_key=True)
    arxiv_id = Column(String(45), primary_key=True)
    title = Column(String(255))
    publication_date = Column(DateTime, default=None)

    citation_count = Column(Integer, default=0)
    pub_url = Column(String(255))

    authors = Column(Text)
    TNCSI = Column(Float, default=None)

    downloaded_pth = Column(Text, default=None)
    categories = Column(String(255))
    valid = Column(Integer, default=0)
    last_update_time = Column(DateTime, default=datetime.now())

    key_idea = Column(Text)
    key_fig = Column(Integer)
     
    paper_type = Column(Integer, default=0)
    authors_title = Column(Text)
    benchmark = Column(Text)
    oa = Column(Text)
    raw_resp = Column(Text)
    search_by_keywords = Column(Text)
    gpt_keyword = Column(Text)
    def __init__(self, arxiv_paper: Arxiv_paper = None,downloaded_pth:str=None,search_by_keywords:str=None):
        if arxiv_paper is not None:
            self.idLiterature = str(uuid.uuid4())   
            self.title = arxiv_paper.title
            self.publication_date = arxiv_paper.publication_date
            self.citation_count = None   
            self.pub_url = arxiv_paper.pub_url
            self.arxiv_id = arxiv_paper.id
            self.authors = '#'.join([str(author) for author in arxiv_paper.authors])
            self.TNCSI = None   
            self.downloaded_pth = downloaded_pth    
            self.categories = '#'.join([str(cat) for cat in arxiv_paper.categories])
            self.valid = -1   
            self.last_update_time = datetime.now()   
            self.key_idea = None   
            self.key_fig = None   
            self.paper_type = None   
            self.authors_title = None   
            self.benchmark = None   
            self.oa = None
            self.raw_resp = None
            self.search_by_keywords = search_by_keywords
        else:
            raise ValueError("A valid Arxiv_paper object must be provided")


class PaperMapping_detail(Base):
    __tablename__ = 'literature_detail'   

    idLiterature_detail = Column(String(36), primary_key=True)
    fig_caps = Column(Text)
    author_details = Column(Text)
    faiss_db_pth = Column(Text)
    table_content = Column(Text)
    abstract = Column(Text)
    language = Column(String(20), default='eng')
    s2_id = Column(String(45))
    publisher = Column(String(255))
    publication_source = Column(String(255))
    source_type = Column(String(255))
    keywords = Column(String(255))
    references = Column(Text)
    comment = Column(Text)
    journal_ref = Column(String(255))
    gpt_keywords = Column(String(255))
    links = Column(Text)
    search_by_keywords = Column(String(255))
    s2_citation_count = Column(Integer)
    s2_publication_date = Column(DateTime, default=None)
    s2_tldr = Column(Text)
    s2_DOI = Column(String(255))
    s2_pub_info = Column(Text)
    s2_pub_url = Column(String(255))
    s2_reference_count = Column(Integer)
    s2_field = Column(String(255))
    s2_influential_citation_count = Column(Integer)
    reference_details = Column(Text)
    authors_num = Column(Integer)
    gpt_keyword = Column(String(255))

    def __init__(self,idLiterature:str,  arxiv_paper: Arxiv_paper = None, s2_paper: S2paper = None,search_by_keywords=None):
        self.fig_caps = None
        self.author_details = None
        self.faiss_db_pth = None
        self.table_content = None
        self.search_by_keywords = search_by_keywords
        self.idLiterature_detail = idLiterature

        if arxiv_paper is not None:
               
            self.abstract = arxiv_paper.abstract
            self.comment =arxiv_paper.comment
            self.journal_ref = arxiv_paper.journal_ref
            self.links = arxiv_paper.links[0].href

        if s2_paper is not None:
            self.s2_id = s2_paper.s2id
            self.s2_tldr = s2_paper.tldr
            self.s2_DOI = s2_paper.DOI
            self.s2_pub_info = s2_paper.publication_source
            self.s2_citation_count = s2_paper.citation_count
            self.s2_reference_count = s2_paper.reference_count
            self.s2_field = s2_paper.field
            self.s2_influential_citation_count = s2_paper.influential_citation_count
            self.authors_num = len(s2_paper.authors)
        self.last_update_time = datetime.now()





class Author(Base):
    __tablename__ = 'author'
    idAuthor = Column(String(36), primary_key=True)
    gs_id = Column(String(64))
    name = Column(String(255))
    affiliation = Column(Text)
    title = Column(Text)
    citation_count = Column(Integer)
    access_date = Column(DateTime, default=datetime.now())
    homepage = Column(Text)

    def __init__(self, name: str, affiliation: str):
        self.idAuthor = str(uuid.uuid4())   
        self.gs_id = None
        self.name = name
        self.affiliation = affiliation
        self.title = None
        self.citation_count = None
        self.homepage = None
        self.access_date = datetime.now()   

class AoP(Base):
    __tablename__ = 'aop'
    idAoP = Column(String(36), primary_key=True)
    idAuthor= Column(String(36))
    idLiterature = Column(String(36))


    def __init__(self, idAuthor:str, idLiterature: str=None):
        self.idAoP =  str(uuid.uuid4())
        self.idAuthor = idAuthor  
        self.idLiterature = idLiterature




Base.metadata.create_all(engine)
