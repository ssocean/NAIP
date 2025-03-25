
import warnings
from datetime import datetime

import arxiv
from retry import retry
from scholarly import scholarly


from retriever.Author import Author
from retriever.Publication import Document

class Google_paper(Document):
    def __init__(self, ref_obj, ref_type='title', **kwargs):
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        self._entity = None if ('entity' not in kwargs) else kwargs['entity']

    @property
    @retry()
    def entity(self):
        if self._entity is None:
            if self.ref_type == 'title':
                rst = scholarly.search_single_pub(self.ref_obj)
                if rst['bib']['title'].lower().replace(" ", "") == self.ref_obj.lower().replace(" ", ""):
                    self._entity = rst
                else:
                    warnings.warn("Haven't fetch anything from google.",
                                  UserWarning)
                    self._entity = False
            return self._entity
        return self._entity

    @property
    def title(self):
        if self.ref_type == 'title':
            return self.ref_obj.lower()
        if self.entity:
            return self.entity['bib']['title'].lower()
        return None

    @property
    def publication_date(self):
        """The data of publication."""
        if not self.entity:
            return None
        if self.entity['bib']['pub_year'] is not None:
            return datetime(year=int(self.entity['bib']['pub_year']), month=1, day=1)
        return None

    @property
    def authors(self):
        """The authors of this document."""
        if self.entity:
            author_list = []
            for google_author in self.entity['bib']['author']:
                author = Author(google_author)
                author_list.append(author)
            return author_list
        return None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        if not self.entity:
            return None
        return self.entity['bib']['venue'] if self.entity['bib']['venue'] else None

    @property
    def abstract(self):
        """The abstract of this document."""
        if not self.entity:
            return None
        return self.entity['bib']['abstract'].replace("\n", "") if self.entity['bib']['abstract'].replace("\n",
                                                                                                          "") else None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        if not self.entity:
            return None
        return self.entity['pub_url'] if 'pub_url' in self.entity else None

    @property
    def citation_count(self):
        if not self.entity:
            return -1
        # try:
        #     print(self.entity['num_citations'])
        # except Exception as E:
        #     print(E)
        #     print(self.entity)
        return self.entity['num_citations'] if 'num_citations' in self.entity else 0

# a = Google_paper('Alleviating pseudo-touching in attention U-Net-based binarization approach for the historical Tibetan document images')
# # a.entity
# print(a.authors)
