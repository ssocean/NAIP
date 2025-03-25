import json
import shelve
import uuid
from urllib.parse import urlencode

import requests
from retry import retry
from scholarly import scholarly
import requests_cache

from CACHE.CACHE_Config import generate_cache_file_name
from config.config import s2api

requests_cache.install_cache('.authorCache', expire_after=36000)


class Author():
    def __init__(self, name, **kwargs):
        self.CACHE_FILE = generate_cache_file_name(force_file_name='.authorsCache')
        self.name = name
        self.kwargs = kwargs
        # print(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.orcid = None if 'orcid' not in self.kwargs else self.kwargs['orcid']
        self._affiliation = None if 'affiliation' not in self.kwargs else self.kwargs['affiliation']
        # self.h_index = None if 'h_index' not in self.kwargs else self.kwargs['h_index']
        # self._affiliation
        # self._h_index = None
        # self._h_index_l5 = None
        # self._i10 = None
        # self._i10_l5 = None
        # self._s2_id = None
        self._entity = None

    def __str__(self):
        # obj_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        # if 'kwargs' in obj_dict:
        #     del obj_dict['kwargs']
        # json_data = json.dumps(obj_dict)
        return str(self.name)

    def __repr__(self):
        obj_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        if 'kwargs' in obj_dict:
            del obj_dict['kwargs']
        json_data = json.dumps(obj_dict)
        return str(json_data)

    @property
    @retry()
    def entity(self):
        if self._entity is None:
            with shelve.open(self.CACHE_FILE) as cache:
                if self.s2_id:
                    if self.s2_id in cache:
                        r = cache[self.s2_id]
                    else:
                        if s2api is not None:
                            headers = {
                                'x-api-key': s2api
                            }
                        else:
                            headers = None
                        r = requests.post(
                            'https://api.semanticscholar.org/graph/v1/author/batch',
                            params={'fields': 'name,hIndex,citationCount,aliases,homepage,affiliations,paperCount,url'},
                            json={"ids": [self.s2_id]},headers=headers
                        )
                        cache[self.s2_id] = r
                    try:
                        self._entity = dict(r.json()[0])
                    except:
                        print(r.json())
                    return self._entity
                else:
                    self._entity = False
        return self._entity

    @property
    def s2_id(self):
        if hasattr(self, '_s2_id'):
            return self._s2_id
        else:
            if self.entity:
                if 'authorId' in self.entity:
                    return self.entity['authorId']
        return None

    @property
    def s2_url(self):
        if hasattr(self, '_s2_url'):
            return self._s2_url
        else:
            if self.entity:
                if 's2_url' in self.entity:
                    return self.entity['url']
        return None

    @property
    def scholar_id(self):
        if self.entity['scholar_id']:
            return self.entity['scholar_id']
        return None

    # @property
    # def s2_id(self):
    #     if self.entity['s2_id']:
    #         return self.entity['s2_id']
    #     return None

    @property
    def affiliations(self):
        if hasattr(self, '_affiliations'):
            if self._affiliations != []:
                return self._affiliations
            return None
        # if self.entity['affiliations'] and self.entity['affiliations'] != '':
        #     return self.entity['affiliations']
        return None

    @property
    def citation_count(self):
        if hasattr(self, '_citationCount'):
            return self._citationCount
        if self.entity:
            if 'citationCount' in self.entity:
                return self.entity['citationCount']
        return 0

    @property
    def paper_count(self):
        if hasattr(self, '_paperCount'):
            return self._paperCount
        if self.entity:
            if 'paperCount' in self.entity:
                return self.entity['paperCount']
        return 0

    @property
    def h_index(self):

        if hasattr(self, '_hIndex'):
            return self._hIndex
        if self.entity:
            if 'hIndex' in self.entity:
                return self.entity['hIndex']
        return -1
    #
    # @property
    # def h_index_l5(self):
    #     if self.entity['h_index_l5']:
    #         return self.entity['h_index_l5']
    #     return None
    #
    # @property
    # def i10(self):
    #     if self.entity['i10']:
    #         return self.entity['i10']
    #     return None
    #
    # @property
    # def i10_l5(self):
    #     if self.entity['i10_l5']:
    #         return self.entity['i10_l5']
    #     return None

    # @property
    # def orcid(self):
    #     if self._orcid is None:
    #         return None if 'orcid' not in self.kwargs else self.kwargs['orcid']
    #
    # @property
    # def affiliation(self):
    #     return None if 'affiliation' not in self.kwargs else self.kwargs['affiliation']
    #
    # @property
    # def h_index(self):
    #     return None if 'h_index' not in self.kwargs else self.kwargs['h_index']
    #
