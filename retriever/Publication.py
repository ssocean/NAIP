"""Stores the metadata of a document.

This is an interface which provides several methods which can be
overridden by child classes. All methods can thus return `None`
in case that method is not overridden.
"""
import re
from abc import ABC, abstractmethod

class Document(ABC):
    def __init__(self, ref_obj, ref_type='title',**kwargs):
        self.ref_obj = ref_obj
        self.ref_type = ref_type
        self._entity = None if ('_entity' not in kwargs) else kwargs['_entity']
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def title(self) -> str:
        if self.ref_type == 'title':
            return self.ref_obj.lower()
    # @property
    # def type(self) -> str:
    #     if self.type is None:
    #         return 'ref'
    #     else:
    #         return self.type
    @property
    def entity(self):
        return None
        # if self._entity is not None:
        #     return self._entity
    @property
    def publication_date(self):
        """The data of publication."""
        return None

    @property
    def id(self):
        """The `DocumentIdentifier` of this document."""
        return None

    @property
    def authors(self):
        """The authors of this document."""
        return None

    @property
    def affiliations(self):
        """The affiliations associated with the authors of this document."""
        authors = self.authors

        if authors is not None:
            items = dict()

            for author in authors:
                for aff in author.affiliations:
                    items[aff.name] = aff

            return list(items.values())

        return None

    @property
    def publisher(self):
        """The publisher of this document."""
        return None

    @property
    def language(self):
        """The language this document is written in."""
        return None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        return None

    @property
    def source_type(self):
        """The type of publication source (i.e., journal, conference
        proceedings, book, etc.)
        """
        return None

    @property
    def keywords(self):
        """The keywords of this document. What exactly consistutes as
        keywords depends on the data source (author keywords, generated
        keywords, topic categories), but is should be a list of strings.
        """
        return None

    @property
    def abstract(self):
        """The abstract of this document."""
        return None

    @property
    def citation_count(self):
        """The number of citations that this document received."""
        return None

    @property
    def references(self):
        """The list of other documents that are cited by this document."""
        return None

    @property
    def citations(self):
        """The list of other documents that cite this document."""
        return None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        return None
    @property
    def field(self):
        """The list of other documents that cite this document."""
        return None
    def __str__(self):
        return (f'Title: {self.title}\t')
    def __repr__(self):
        return (f'{(self.title)}\t')
    # def persistance_to_database(self):

    def mentions(self, term: str) -> bool:
        """Returns `True` if this document mentions the given term in the
        title, abstract, or keywords.
        """
        pattern = r"(^|\s)" + re.escape(term) + r"($|\s)"
        flags = re.IGNORECASE
        keywords = self.keywords or []

        for text in [self.title, self.abstract] + keywords:
            if text and re.search(pattern, text, flags=flags):
                return True

        return False
