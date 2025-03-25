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
from tools.Reference import filter_punctuation, get_paper_info_from_REST


class Crossref_paper(Document):
    def __init__(self, ref_obj, ref_type='title', **kwargs):
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        self._entity = None if ('entity' not in kwargs) else kwargs['entity']

    @property
    @retry()
    def entity(self, max_tries=5):
        if self._entity is None:
            if self.ref_type == 'title':
                rst = get_paper_info_from_REST(self.ref_obj)
                for i, matched_paper in enumerate(rst['message']['items']):
                    if matched_paper['title'][0].lower().replace(" ", "") == self.ref_obj.lower().replace(
                            " ", ""):
                        self._entity = matched_paper
                        return self._entity
                    if i >= max_tries or i == len(rst['message']['items']) - 1:
                        warnings.warn("Haven't fetch anything from crossref. ",
                                      UserWarning)
                        self._entity = False
                        return self._entity
        return self._entity

    @property
    def title(self):

        if self.entity:
            return self.entity['title'][0].lower()
        if self.ref_type == 'title':
            return self.ref_obj.lower()

    @property
    def publication_date(self):
        """The data of publication."""
        if self.entity['created']['date-time'] is not None:
            return datetime.strptime(self.entity['created']['date-time'], "%Y-%m-%dT%H:%M:%SZ")
        return None

    @property
    def id(self):
        """The `DocumentIdentifier` of this document."""
        return self.entity['DOI'] if self.entity['DOI'] else None

    @property
    def authors(self):
        """The authors of this document."""
        if self.entity:
            author_list = []
            for crossref_author in self.entity['author']:
                author = Author(crossref_author['given']+' '+crossref_author['family'])
                if 'ORCID' in crossref_author and crossref_author['ORCID'] != '':
                    author.orcid = crossref_author['ORCID']
                if 'affiliation' in crossref_author and crossref_author['affiliation'] != []:
                    author._affiliation = crossref_author['affiliation']
                author_list.append(author)
            return author_list
        return None

    @property
    def affiliation(self):
        return self.authors[0].affiliation
    @property
    def publisher(self):
        """The publisher of this document."""
        return self.entity['publisher'] if self.entity['publisher'] else None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        return self.entity['container-title'][0] if self.entity['container-title'][0] else None


    @property
    def abstract(self):
        """The abstract of this document."""
        return self.entity["abstract"].replace("\n", "") if self.entity["abstract"].replace("\n", "") else None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        return self.entity['primary']['URL'] if self.entity['primary']['URL'] else None

    @property
    def citation_count(self):

        return self.entity['is-referenced-by-count'] if self.entity['is-referenced-by-count'] else None


