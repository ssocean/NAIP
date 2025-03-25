import random
import time
import warnings
from datetime import datetime

import arxiv
from retry import retry
from scholarly import scholarly
# OPENAI SETUP
import openai
import requests
import pandas as pd
import matplotlib.pyplot as plt
import string

from retriever.Author import Author
from retriever.Publication import Document

import uuid
import mysql.connector

def filter_punctuation(title):
     
    punctuation = string.punctuation

     
    translator = str.maketrans("", "", punctuation)

     
    filtered_title = title.translate(translator)

    return filtered_title

def get_arxiv_id_from_url(url:str):
    if url.endswith('.pdf'):
        url = url[:-4]
    id = url.split(r'/')[-1]
    return id




class Arxiv_paper(Document):
    def __init__(self,ref_obj, ref_type='title', **kwargs):
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        # self._entity = None

    # @retry()
    @property
    def entity(self,max_tries=5):
        if self._entity is None:
            if self.ref_type == 'title':
                search = arxiv.Search(query=f'ti:{filter_punctuation(self.ref_obj)}')
                for i,matched_paper in enumerate(search.results()):
                    if matched_paper.title.lower().replace(" ", "") == self.ref_obj.lower().replace(" ", ""):
                        self._entity = matched_paper
                        return self._entity
                    if i>=max_tries:
                        warnings.warn(
                            "Haven't fetch anything from arxiv. ",
                            UserWarning)
                        self._entity = False
                        return self._entity
            elif self.ref_type == 'id':
                # print('searching id')
                search = arxiv.Search(id_list=[self.ref_obj])
                # print('searching done')
                matched_paper = next(search.results())
                # print('fetching Done')
                self._entity = matched_paper
                # if matched_paper.entry_id == self.ref_obj:
                #     self._entity = matched_paper
                # else:
                #     warnings.warn(
                #         "Haven't fetch anything from arxiv.",
                #         UserWarning)
                #     self._entity = False
                return self._entity
            elif self.ref_type == 'entity':
                self._entity = self.ref_obj
                return self._entity
        return self._entity


    @property
    def title(self):
        if self.ref_type == 'title':
            return self.ref_obj
        if self.entity:
            return self.entity.title
        return None
    @property
    def publication_date(self):
        """The data of publication."""
        return self.entity.published if self.entity.published else None

    @property
    def id(self):
        """The `DocumentIdentifier` of this document."""
        return self.entity.entry_id if self.entity.entry_id else None

    @property
    def authors(self):
        """The authors of this document."""
        if self.entity:
            return [Author(i.name) for i in self.entity.authors]
        return None



    @property
    def publisher(self):
        """The publisher of this document."""
        return 'arxiv'



    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        return 'arxiv'

    @property
    def source_type(self):
        """The type of publication source (i.e., journal, conference
        proceedings, book, etc.)
        """
        return 'pre-print'


    @property
    def abstract(self):
        """The abstract of this document."""
        return self.entity.summary if self.entity.summary else None



    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        return self.entity.entry_id if self.entity.entry_id else None

    @property
    def comment(self):
        '''The authors comment if present. '''
        return self.entity.comment if self.entity.comment else None
    @property
    def journal_ref(self):
        '''A journal reference if present. '''
        return self.entity.journal_ref if self.entity.journal_ref else None
    @property
    def primary_category(self):
        '''The primary arXiv category. '''
        return self.entity.primary_category if self.entity.primary_category else None
    @property
    def categories(self):
        '''The arXiv or ACM or MSC category for an article if present.'''
        return self.entity.categories if self.entity.categories else None
    @property
    def links(self):
        '''Can be up to 3 given url's associated with this article. '''
        return self.entity.links if self.entity.links else None

# a = Arxiv_paper('segment anything')
# # a.entity
# print(a)
# print(a.authors)
# def my_function(name,**kwargs):
#     print(name)
 
#     for key, value in kwargs.items():
#         print(key, value)
# my_function(name='Alice', age=25, city='New York')