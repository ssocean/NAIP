import re
from retriever.Author import Author
from retriever.Publication import Document
import logging
import math
import statistics
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from config.config import s2api
from tools.gpt_util import get_chatgpt_field


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from retry import retry


class S2paper(Document):
    def __init__(self, ref_obj, ref_type='title', filled_authors=True, force_return=False, use_cache=True, **kwargs):
        '''

        :param ref_obj: search keywords
        :param ref_type: title   entity
        :param filled_authors:  retrieve detailed info about authors?
        :param force_return:  even title is not mapping, still return the result
        :param kwargs:
        '''
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        # Expectation: A typical program is unlikely to create more than 5 of these.
        self.S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
        self.S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.CACHE_FILE = ".ppicache"
        self.DEFAULT_TIMEOUT = 3.05  # 100 requests per 5 minutes
        self._entity = None
        # a few settings
        self.filled_authors = filled_authors
        self.force_return = force_return
        self._gpt_keyword = None
        self._gpt_keywords = None
        self._TNCSI = None
        self._IEI = None
        self._RQM = None
        self._RUI = None
        self.use_cache = use_cache

    @property
    @retry()
    def entity(self, max_tries=5):
        if self.ref_type == 'entity':
            self._entity = self.ref_obj
            return self._entity

        if self._entity is None:
            url = f"{self.S2_QUERY_URL}?query={self.ref_obj}&fieldsOfStudy=Computer Science&fields=url,title,abstract,authors,venue,externalIds,referenceCount,tldr,openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,publicationVenue&offset=0&limit=1"
            with shelve.open(generate_cache_file_name(url)) as cache:

                # params = urlencode(dict(query=self.ref_obj, offset=0, limit=1))
                # params = quote_plus(self.ref_obj)
                # print(params)
                # fieldsOfStudy=Computer Science&
                if url in cache and self.use_cache and cache[url].status_code==200:
                    reply = cache[url]
                else:
                    session = requests.Session()
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    reply = session.get(url, headers=headers)
                    cache[url] = reply

            response = reply.json()
            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                self._entity = False
                return self._entity
                # raise Exception(f"error while fetching {reply.url}: {msg}")
            else:

                if self.ref_type == 'title' and re.sub(r'\W+', '', response['data'][0]['title'].lower()) != re.sub(
                        r'\W+', '', self.ref_obj.lower()):
                    if self.force_return:
                        self._entity = response['data'][0]
                        return self._entity
                    else:
                        print(response['data'][0]['title'].lower())
                        self._entity = False
                        return self._entity
                else:
                    self._entity = response['data'][0]
                    return self._entity
        return self._entity

    @property
    def gpt_keyword(self):
        # only one keyword is generated
        if self._gpt_keyword is None:
            self._gpt_keyword = get_chatgpt_field(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keyword

    @property
    def gpt_keywords(self):
        # get multiple keyword at one time
        if self._gpt_keywords is None:
            self._gpt_keywords = get_chatgpt_fields(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keywords

    @property
    def title(self):
        if self.ref_type == 'title':
            return self.ref_obj.lower()
        if self.entity:
            return self.entity.get('title').lower()
        return None

    @property
    def publication_date(self):
        """The data of publication."""
        if self.entity:
            # if 'publicationDate' in self.entity and self.entity['publicationDate'] is not None:
            if self.entity.get('publicationDate') is not None:
                return datetime.strptime(self.entity['publicationDate'], "%Y-%m-%d")
        return None

    @property
    def s2id(self):
        """The `DocumentIdentifier` of this document."""
        return self.entity['paperId'] if self.entity else None

    @property
    def tldr(self):
        """The `DocumentIdentifier` of this document."""
        if self.entity:
            # if 'tldr' in self.entity and self.entity['tldr'] is not None:
            if self.entity.get('tldr') is not None:
                return self.entity['tldr']['text']
        return None

    @property
    def DOI(self):
        if self.entity:
            # if 'DOI' in self.entity['externalIds'] and self.entity['externalIds']['DOI'] is not None:
            return self.entity.get('DOI')  # is not None:
            # return self.entity['externalIds']['DOI']
        return None

    @property
    @retry()
    def authors(self):
        """The authors of this document."""
        if self.entity:
            authors = []
            if 'authors' in self.entity:
                if not self.filled_authors:
                    for item in self.entity['authors']:
                        author = Author(item['name'], _s2_id=item['authorId'])
                        # author.entity
                        authors.append(author)
                    return authors
                else:
                    url = (f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/authors?fields=authorId,'
                           f'externalIds,name,affiliations,homepage,paperCount,citationCount,hIndex,url')

                    with shelve.open(generate_cache_file_name(url)) as cache:
                        if url in cache and self.use_cache and cache[url].status_code == 200:
                            r = cache[url]
                        else:
                            if s2api is not None:
                                headers = {
                                    'x-api-key': s2api
                                }
                            else:
                                headers = None
                            r = requests.get(
                                url,
                                headers=headers
                            )
                            r.raise_for_status()
                        for item in r.json()['data']:
                            author = Author(item['name'], _s2_id=item['authorId'], _s2_url=item['url'],
                                            _h_index=item['hIndex'], _citationCount=item['citationCount'],
                                            _paperCount=item['paperCount'])
                            authors.append(author)
                        return authors

        return None

    @property
    def affiliations(self):
        if self.authors:
            affiliations = []
            for author in self.authors:
                if author.affiliations is not None:
                    affiliations.append(author.affiliations)
            return ';'.join(list(set(affiliations)))
        return None

    @property
    def publisher(self):
        """The publisher of this document."""
        if self.entity:
            return self.entity.get('publicationVenue')
            # if 'publicationVenue' in self.entity and self.entity['publicationVenue'] is not None:
            #     return self.entity['publicationVenue']
        return None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        if self.entity:
            # if 'venue' in self.entity and self.entity['venue'] is not None:
            #     return self.entity['venue']
            return self.entity.get('venue')
        return None

    @property
    def source_type(self):
        if self.entity:
            # if 'publicationTypes' in self.entity and self.entity['publicationTypes'] is not None:
            #     return self.entity['publicationTypes']
            return self.entity.get('publicationTypes')
        return None

    @property
    def abstract(self):
        """The abstract of this document."""
        if self.entity:
            # if 'abstract' in self.entity and self.entity['abstract'] is not None:
            #     return self.entity['abstract']
            return self.entity.get('abstract')
        return None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        if self.entity:
            # if 'openAccessPdf' in self.entity:
            #     return self.entity['openAccessPdf']
            return self.entity.get('openAccessPdf')
        return None

    @property
    def citation_count(self):

        if self.entity:
            # if 'citationCount' in self.entity:
            #     return self.entity['citationCount']
            return self.entity.get('citationCount')
        return None

    @property
    def reference_count(self):

        if self.entity:
            # if 'citationCount' in self.entity:
            #     return self.entity['referenceCount']
            return self.entity.get('referenceCount')
        return None

    @property
    def field(self):
        if self.entity:
            if self.entity.get('s2FieldsOfStudy') is not None:
                fields = []
                for fdict in self.entity.get('s2FieldsOfStudy'):
                    category = fdict['category']
                    fields.append(category)
                fields = ','.join(list(set(fields)))
                return fields
        return None

    @property
    def influential_citation_count(self):
        if self.entity:
            # if 'influentialCitationCount' in self.entity:
            #     return self.entity['influentialCitationCount']
            return self.entity.get('influentialCitationCount')
        return None

    def plot_s2citaions(keyword: str, year: str = '2018-2023', total_num=100, CACHE_FILE='.semantischolar'):
        l = 0
        citation_count = []
        influentionCC = []

        return citation_count, influentionCC

    @property
    @retry()
    def references(self):
        if self.entity:
            references = []
            url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/references?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=999'

            with shelve.open(generate_cache_file_name(url)) as cache:
                # print(url) #references->citations
                if url in cache:
                    r = cache[url]
                else:
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    r = requests.get(url, headers=headers)

                    r.raise_for_status()

                    cache[url] = r
                if 'data' not in r.json():
                    return []
                for item in r.json()['data']:
                    # print(item)
                    ref = S2paper(item['citedPaper']['title'])
                    ref.filled_authors = False
                    info = {'paperId': item['citedPaper']['paperId'], 'contexts': item['contexts'],
                            'intents': item['intents'], 'isInfluential': item['isInfluential'],
                            'title': item['citedPaper']['title'], 'venue': item['citedPaper']['venue'],
                            'citationCount': item['citedPaper']['citationCount'],
                            'influentialCitationCount': item['citedPaper']['influentialCitationCount'],
                            'publicationDate': item['citedPaper']['publicationDate']}
                    # authors = []

                    ref._entity = info
                    # print(ref.citation_count)
                    references.append(ref)
                return references

        return None

    @property
    @retry()
    def citations_detail(self):
        if self.entity:
            references = []
            is_continue = True
            offset = 0
            while is_continue:

                url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/citations?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=1000&offset={offset}'
                offset += 1000
                with shelve.open(generate_cache_file_name(url)) as cache:
                    # print(url) #references->citations
                    if url in cache:
                        r = cache[url]
                    else:
                        if s2api is not None:
                            headers = {
                                'x-api-key': s2api
                            }
                        else:
                            headers = None
                        r = requests.get(url, headers=headers)
                        r.raise_for_status()
                        cache[url] = r
                    if 'data' not in r.json() or r.json()['data'] == []:
                        is_continue = False
                    for item in r.json()['data']:
                        # print(item)
                        ref = S2paper(item['citingPaper']['title'])

                        ref.filled_authors = True

                        info = {'paperId': item['citingPaper']['paperId'], 'contexts': item['contexts'],
                                'intents': item['intents'], 'isInfluential': item['isInfluential'],
                                'title': item['citingPaper']['title'], 'venue': item['citingPaper']['venue'],
                                'citationCount': item['citingPaper']['citationCount'],
                                'influentialCitationCount': item['citingPaper']['influentialCitationCount'],
                                'publicationDate': item['citingPaper']['publicationDate'],
                                'authors': item['citingPaper']['authors']}
                        # authors = []

                        ref._entity = info
                        # print(ref.citation_count)
                        references.append(ref)
            return references

        return None
    @property
    # @retry()
    def TNCSI(self):
        if self._TNCSI is None:

            self._TNCSI = get_TNCSI(self,topic_keyword=self.gpt_keyword,show_PDF=False)
        return self._TNCSI

    @property
    @retry()
    def IEI(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._IEI is None:
                self._IEI = get_IEI(self.title, normalized=True, exclude_last_n_month=3)
            return self._IEI
        return float('-inf')

    @property
    @retry()
    def RQM(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RQM is None:
                self._RQM = get_RQM(self, tncsi_rst=self.TNCSI, beta=5)
            return self._RQM
        return float('-inf')
    @property
    @retry()
    def RUI(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RUI is None:
                self._RUI = get_RUI(self)
            return self._RUI
        return float('-inf')


# sp = S2paper('segment anything')

# print(sp.references)
# print(sp.influential_citation_count)
S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"~\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"



def relevance_query(query, pub_date: datetime = None):
    '''
    :param query: keyword
    :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
    :param pub_date:
    2019-03-05 on March 3rd, 2019
    2019-03 during March 2019
    2019 during 2019
    2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
    1981-08-25: on or after August 25th, 1981
    :2015-01 before or on January 31st, 2015
    2015:2020 between January 1st, 2015 and December 31st, 2020
    :return:
    '''
    s2api = None
    p_dict = dict(query=query)
    if pub_date:
        p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'

    params = urlencode(p_dict)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?{params}&fields=url,title,abstract,authors,venue,referenceCount,"
        f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
        f"s2FieldsOfStudy,publicationTypes,publicationDate")
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        if url in cache:
            reply = cache[url]
        else:
            session = requests.Session()
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }
            else:
                headers = None
            reply = session.get(url, headers=headers)
            cache[url] = reply

            reply = session.get(url)
        response = reply.json()

        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            raise Exception(f"error while fetching {reply.url}: {msg}")

        return response



def request_query(query, sort_rule=None, pub_date: datetime = None, continue_token=None):
    '''
    :param query: keyword
    :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
    :param pub_date:
    2019-03-05 on March 3rd, 2019
    2019-03 during March 2019
    2019 during 2019
    2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
    1981-08-25: on or after August 25th, 1981
    :2015-01 before or on January 31st, 2015
    2015:2020 between January 1st, 2015 and December 31st, 2020
    :return:
    '''
    s2api = None
    p_dict = dict(query=query)
    if pub_date:
        p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'
    if continue_token:
        p_dict['token'] = continue_token
    if sort_rule:
        p_dict['sort'] = sort_rule
    params = urlencode(p_dict)
    url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    # print(url)
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        rt = False
        if url in cache:
            reply = cache[url]
            try:
                response = reply.json()
            except:
                rt = True
        if url not in cache or rt:
            session = requests.Session()
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }
            else:
                headers = None
            reply = session.get(url, headers=headers)

            response = reply.json()
            cache[url] = reply

        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            raise Exception(f"error while fetching {reply.url}: {msg}")

        return response

# @retry()
def get_s2citaions_per_month(title, total_num=2000):
    '''

    :param keyword: topic keyword
    :param year: like 2018-2023 || 2018
    :param total_num:  fetching up to total_num results
    :param CACHE_FILE:
    :return:
    '''
    s2paper = S2paper(title)
    if s2paper.publication_date is None:
        print('NO PUBLICATION DATE RECORDER')
        return []
    s2id = s2paper.s2id
    # print(s2id)
    citation_count = {}
    missing_count = 0
    OFFSET = 0

    url_temp = f'https://api.semanticscholar.org/graph/v1/paper/{s2id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset'
    with shelve.open(generate_cache_file_name(url_temp)) as cache:
        for i in range(int(total_num / 1000)):
            url = f'https://api.semanticscholar.org/graph/v1/paper/{s2id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset={OFFSET}&limit=1000'
            # url = f'https://api.semanticscholar.org/graph/v1/paper/{s2id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset={1000 * i}&limit=1000'

            OFFSET+=1000
            if url in cache:

                r = cache[url]
            else:
                # print('retrieving')
                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                r = requests.get(url, headers=headers,verify=True)
                r.raise_for_status()
                # time.sleep(0.5)
                cache[url] = r

            if 'data' not in r.json():
                return []
            # print(r.json()['data'])
            if r.json()['data'] == []:
                break
            for item in r.json()['data']:
                # print(item)

                info = {'paperId': item['citingPaper']['paperId'], 'title': item['citingPaper']['title'],
                        'citationCount': item['citingPaper']['citationCount'],
                        'publicationDate': item['citingPaper']['publicationDate']}
                # authors = []

                ref = S2paper(info, ref_type='entity')
                ref.filled_authors = False
                try:
                    if s2paper.publication_date <= ref.publication_date and ref.publication_date <= datetime.now():
                        dict_key = str(ref.publication_date.year) + '.' + str(ref.publication_date.month)
                        citation_count.update({dict_key: citation_count.get(dict_key, 0) + 1})
                    else:
                        missing_count += 1
                except Exception as e:
                    # print(e)
                    # print(ref.publication_date)
                    # print('No pub time')
                    missing_count += 1
                    continue
    # print(f'Missing count:{missing_count}')
     
    # print(citation_count)

    sorted_data = OrderedDict(
        sorted(citation_count.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True))
    # print(f'{s2id} missing {missing_count} due to abnormal info.')
     
    latest_month = datetime.now()#.strptime(s2paper.publication_date, "%Y.%m")# datetime.now().strftime("%Y.%m")
    earliest_month = s2paper.publication_date#.strftime("%Y.%m")#datetime.strptime(s2paper.publication_date, "%Y.%m")

     
    all_months = [datetime.strftime(latest_month, "%Y.%#m")]
    while latest_month > earliest_month:
        latest_month = latest_month.replace(day=1)   
        latest_month = latest_month - timedelta(days=1)   
        all_months.append(datetime.strftime(latest_month, "%Y.%#m"))

     
    result = {month: sorted_data.get(month, 0) for month in all_months}
     
    result = OrderedDict(sorted(result.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True))
    # print(dict(result))
    return result
def _get_TNCSI_score(citation:int, loc, scale):
    import math
    def exponential_cdf(x, loc, scale):
        if x < loc:
            return 0
        else:
            z = (x - loc) / scale
            cdf = 1 - math.exp(-z)
            return cdf

    # print(citation, loc, scale)
    aFNCSI = exponential_cdf(citation, loc, scale)
    return aFNCSI
def fit_topic_pdf(topic, topk=2000, show_img=False, save_img_pth=None):
    import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image
    import io




    citation, _ = _plot_s2citaions(topic, total_num=1000)
    citation = np.array(citation)

    try:
        params = stats.expon.fit(citation)
    except:
        # print(citation)
        return None,None
    loc, scale = params
    if len(citation)<=1:
        return None, None
     
    x = np.linspace(np.min(citation), np.max(citation), 100)
    pdf = stats.expon.pdf(x, loc, scale)

     
    if show_img or save_img_pth:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.hist(citation, bins=1000, density=True, alpha=0.5)
        plt.plot(x, pdf, 'r', label='Fitted Exponential Distribution')
        plt.xlabel('Number of Citations')
         
        plt.ylabel('Frequency')
        plt.legend()
        if save_img_pth:
            print('saving success')
            plt.savefig(save_img_pth)
        plt.show()


    return loc, scale  # , image
@retry(delay=6)
def _plot_s2citaions(keyword: str, year: str = None, total_num=1000, CACHE_FILE='.ppicache'):
    '''

    :param keyword: topic keyword
    :param year: like 2018-2023 || 2018
    :param total_num:  fetching up to total_num results
    :param CACHE_FILE:
    :return:
    '''
    l = 0
    citation_count = []
    influentionCC = []

    for i in range(int(total_num // 100)):
        if year:
            url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&year={year}&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'
        else:
            url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'
        with shelve.open(generate_cache_file_name(url)) as cache:
            if url in cache:
                r = cache[url]
            else:

                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                r = requests.get(url, headers=headers,verify=True)
                if 'Requested data for this limit and/or offset is not available' in r.text:
                    continue
                r.raise_for_status()

                time.sleep(0.5)
                cache[url] = r


        # print(r.json())

        try:
            if 'data' not in r.json():
                # logger.info(f'Fetching {l} data from SemanticScholar.')
                return citation_count, influentionCC
                # raise ConnectionError

            for item in r.json()['data']:
                if int(item['citationCount']) >= 0:
                    citation_count.append(int(item['citationCount']))
                    influentionCC.append(int(item['influentialCitationCount']))
                    l += 1
                else:
                    print(item['citationCount'])
        except:
            continue

        # logger.info(f'Fetch {l} data from SemanticScholar.')
    return citation_count, influentionCC
@retry(tries=3)
def get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False):
    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None
    if s2paper.citation_count is None:
        rst = {}
        rst['TNCSI'] = -1
        rst['topic'] = 'NONE'
        return rst
    if topic_keyword is None:

        topic = get_chatgpt_field(ref_obj, s2paper.abstract)
        topic = topic.replace('.', '')
        logger.info(
            f'Generated research field is {topic}.')
    else:
        topic = topic_keyword
        # logger.info(f'Pre-defined research field is {topic}')

    loc, scale = fit_topic_pdf(topic, topk=2000,save_img_pth=save_img_pth,show_img=show_PDF)
    if loc is not None and scale is not None:
        try:
            TNCSI = _get_TNCSI_score(s2paper.citation_count, loc, scale)
        except ZeroDivisionError as e:
            rst = {}
            rst['TNCSI'] = -1
            rst['topic'] = 'NaN'
            return rst
        rst = {}
        rst['TNCSI'] = TNCSI
        rst['topic'] = topic
        rst['loc'] = loc
        rst['scale'] = scale

        return rst
    else:
        rst = {}
        rst['TNCSI'] = -1
        rst['topic'] = topic
        rst['loc'] = loc
        rst['scale'] = scale
        return rst


from retry import retry
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

@retry()
def get_IEI(title, show_img=False, save_img_pth=None,exclude_last_n_month=1,normalized=False):
    spms = get_s2citaions_per_month(title, 2000)
    actual_len = 6 if len(spms) >= 6+exclude_last_n_month else len(spms) -exclude_last_n_month
    # print(f'acutal len:{actual_len}')
    if actual_len < 6:
        rst = {}
        rst['L6'] = float('-inf')
        rst['I6'] = float('-inf')
        return rst
     
    x = [i for i in range(actual_len)]
    subset = list(spms.items())[exclude_last_n_month:exclude_last_n_month+actual_len][::-1]
    y = [item[1] for item in subset]
    if normalized:
        y = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
     
    t = np.linspace(0, 1, 100)
    n = len(x) - 1   
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)
    # print(n,y)

    for i in range(n + 1):
        curve_x += comb(n, i) * (1 - t) ** (n - i) * t ** i * x[i]
        curve_y += comb(n, i) * (1 - t) ** (n - i) * t ** i * y[i]
    if show_img or save_img_pth:
         
        plt.clf()
        fig = plt.figure(figsize=(6, 4), dpi=300)  # Increase DPI for high resolution
        plt.style.use('seaborn-v0_8')
        plt.plot(x, y, 'o', color='darkorange', label='Data Point')  # darkorange for contrast
        plt.plot(curve_x, curve_y, color='steelblue', label='Bezier Curve')  # steelblue for the line

        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('Received Citation')
        # plt.title('Quintic Bezier Curve')
        plt.grid(True)
        if save_img_pth:
            plt.savefig(save_img_pth,dpi=300, bbox_inches='tight')
        if show_img:
            plt.show()

    dx_dt = np.zeros_like(t)
    dy_dt = np.zeros_like(t)
    # print(y)
    for i in range(n):
        dx_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (x[i + 1] - x[i])
        dy_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (y[i + 1] - y[i])

    I6 = dy_dt[-1] / dx_dt[-1]
    # print(len(dy_dt))
    slope_avg = []
    # sum([dy_dt[i-1] / dx_dt[i-1] for i in range(0, 101, 20)])/n+1
    for i in range(0, 100, 20):
        # print((curve_x[i-1],curve_y[i-1]))
        slope_avg.append(dy_dt[i] / dx_dt[i])
    slope_avg.append(I6)
    print(slope_avg)
    print('平均', sum(slope_avg) / 6)
    print("瞬时斜率:", I6)
    rst = {}
    rst['L6'] = sum(slope_avg) / 6 if not math.isnan(sum(slope_avg)) else float('-inf')
    rst['I6'] = I6 if not math.isnan(I6) else float('-inf')
    return rst



S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"~\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"


from CACHE.CACHE_Config import generate_cache_file_name
from tools.gpt_util import get_chatgpt_fields

import requests
from urllib.parse import urlencode
import shelve

S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"~\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"
@retry()
def request_query(query,  sort_rule=None,  continue_token=None, early_date: datetime = None, later_date:datetime = None
                  ):# before_pub_date=True
    '''

    :param query:
    :param offset:
    :param limit:
    :param CACHE_FILE:
    :param sort: publicationDate:asc - return oldest papers first.
                citationCount:desc - return most highly-cited papers first.
                paperId - return papers in ID order, low-to-high.
    :param pub_date:
    2019-03-05 on March 3rd, 2019
    2019-03 during March 2019
    2019 during 2019
    2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
    1981-08-25: on or after August 25th, 1981
    :2015-01 before or on January 31st, 2015
    2015:2020 between January 1st, 2015 and December 31st, 2020
    :return:
    '''
    s2api = None
    p_dict = dict(query=query)

    if early_date and later_date is None:
        p_dict['publicationDateOrYear'] = f'{early_date.strftime("%Y-%m-%d")}:'
    elif later_date and early_date is None:
        p_dict['publicationDateOrYear'] = f':{later_date.strftime("%Y-%m-%d")}'
    elif later_date and early_date:
        p_dict['publicationDateOrYear'] = f'{early_date.strftime("%Y-%m-%d")}:{later_date.strftime("%Y-%m-%d")}'
    else:
        pass
        # if before_pub_date:
        #
        # else:
        #
    if continue_token:
        p_dict['token'] = continue_token
    if sort_rule:
        p_dict['sort'] = sort_rule
    params = urlencode(p_dict)
    url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    # print(url)
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        if url in cache:
            reply = cache[url]
            if reply.status_code == 504:
                session = requests.Session()
                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                reply = session.get(url, headers=headers)
                cache[url] = reply

                reply = session.get(url)
        else:
            session = requests.Session()
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }
            else:
                headers = None
            reply = session.get(url, headers=headers)
            cache[url] = reply

            reply = session.get(url)
        response = reply.json()

        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            raise Exception(f"error while fetching {reply.url}: {msg}")

        return response

def get_Coverage(ref_obj, ref_type='entity', tncsi_rst=None, multiple_keywords=False):
    def cal_weighted_cocite_rate(ref_relevant_lst, pub_relevant_lst,tncsi_rst):
        '''

        :param ref_list_Anchor:
        :param ref_list_eval:
        :return:
        '''
        loc = tncsi_rst['loc']
        scale = tncsi_rst['scale']
        ref_list = [(i.title,i.citation_count) for i in ref_relevant_lst]
        pub_list = [(i.title,i.citation_count) for i in pub_relevant_lst]

        # print(len(ref_list))
        # print(len(pub_list))
        ref_set = set(ref_list)
        pub_set = set(pub_list)
        # print()
        # print(len(ref_list))
        # print(len(pub_set))

        intersection = ref_set & pub_set
        intersection = list(intersection)
        # print(anchor_set)
        #
        #
        # print(eval_set)

        # print(len(intersection))

        # exclude = ref_set - set(intersection)
        # print(exclude)
        # print(len(exclude))
        # print(intersection)
        score = 0
        print(f'raw coverage:{len(intersection)/len(pub_set)}')
        for item in intersection:
            score += _get_TNCSI_score(item[-1],loc,scale)#1
        try:
            score_of_relevant = sum([_get_TNCSI_score(i[-1],loc,scale) for i in list(pub_set)])
            overlap_ratio = score / score_of_relevant#len(pub_set)
        except ZeroDivisionError:
            return 0
        return overlap_ratio

    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if tncsi_rst:
        loc = tncsi_rst['loc']
        scale = tncsi_rst['scale']
    else:
        print('TODO')
        return

    C = int(2 * (math.log(0.01, math.e) * (-1 * scale) + loc))

    # get Reference_relevant
    ref_r = s2paper.references
    print(f'search paper title:{s2paper.title}, which has {len(ref_r)} refs')
    sort_rule = 'citationCount:desc'  # 'citationCount:desc' 'publicationDate:desc'

    # get Publications_relevant
    pub_r = []
    continue_token = None
    if multiple_keywords:
        keywords = s2paper.gpt_keywords
        # keywords = ['Instance segmentation','panoptic segmentation','semantic segmentation','weakly supervised','unsupervised','domain adaptation']
        # keywords = ['curriculum learning', 'self-paced learning', 'training strategy']
        keywords = [f'"{i}"~25' for i in keywords]
        print(keywords)
        topic = ' | '.join(keywords)
        for i in range(0, C, 1000):
            if continue_token is None:
                response = request_query(topic, sort_rule=sort_rule, pub_date=s2paper.publication_date)
            else:
                response = request_query(topic, sort_rule=sort_rule, pub_date=s2paper.publication_date,
                                         continue_token=continue_token)
            # print(response.keys())

            if "token" in response:
                continue_token = response['token']

            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                logger.warning('No matched paper!')
                # raise Exception(f"error while fetching {reply.url}: {msg}")
            else:
                for entity in response['data']:
                    temp_ref = S2paper(entity, ref_type='entity', force_return=False, filled_authors=False)
                    pub_r.append(temp_ref)

    else:
        for i in range(0, C, 1000):
            if continue_token is None:
                response = request_query(tncsi_rst['topic'], sort_rule=sort_rule, pub_date=s2paper.publication_date)
            else:
                response = request_query(tncsi_rst['topic'], sort_rule=sort_rule, pub_date=s2paper.publication_date,
                                         continue_token=continue_token)
            # print(response.keys())
            if "token" in response:
                continue_token = response['token']
            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                logger.warning('No matched paper!')
                # raise Exception(f"error while fetching {reply.url}: {msg}")
            else:
                for entity in response['data']:
                    temp_ref = S2paper(entity, ref_type='entity', force_return=False, filled_authors=False)
                    pub_r.append(temp_ref)

    print(f'Success retrieving {len(pub_r)}/{C} related work, ')

    # print(len(pub_r))

    coverage = cal_weighted_cocite_rate(ref_r, pub_r[:C],tncsi_rst)
    return coverage



from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def get_median_pubdate(pub_time,refs):
    pub_dates = []
    for i in refs:
        if i.publication_date:
            pub_dates.append(i.publication_date)
    # pub_dates = [i.publication_date for i in s2paper.references]
    sorted_list = sorted(pub_dates, reverse=True)
    index = 0
    while index < len(sorted_list):
        try:
            sorted_list[index].timestamp()
            index += 1
        except Exception as e:
            print(f"Error: {e}")
            del sorted_list[index]

    timestamps = [d.timestamp() for d in sorted_list]
    if len(timestamps) == 0:
        return float('-inf')
    median_timestamp = statistics.median(timestamps)
    median_value = datetime.fromtimestamp(median_timestamp)

    # median_value = statistics.median(sorted_list)
    # months_difference = (pub_time - median_value) // datetime.timedelta(days=30)
    return median_value
 

def plot_time_vs_aFNCSI(sp: S2paper, loc, scale):
    def create_cool_colors():
        colors = ["#D6EDFF", "#B1DFFF", "#8CCBFF", "#66B8FF", "#FF69B4",
                  "#1D8BFF", "#0077E6", "#005CBF", "#00408C", "#002659"]

        return random.choice(colors)

    def create_warm_colors():
        colors = ["#FFD6D6", "#FFB1B1", "#FF8C8C", "#FF6666", "#FF4040",
                  "#FF1D1D", "#E60000", "#BF0000", "#8C0000", "#590000"]

        return random.choice(colors)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

     
    cool_cmap = create_cool_colors()

     
    warm_cmap = create_warm_colors()
    times = []
    aFNCSIs = []
    areas = []
    sp_pub_date = sp.publication_date
    for i in tqdm(sp.references):

        ref_time = len(i._entity['contexts'])
        # importance = min(math.log10(ref_time + 1),1)
        if i.publication_date:
            pub_date = i.publication_date
            cite_count = 0 if i.citation_count is None else int(i.citation_count)
            cur_aFNCSI = _get_TNCSI_score(cite_count, loc, scale)
            # ref_time = len(i._entity['contexts'])
            # importance = min(math.log10(ref_time + 1),1)
            # icite_count = 0 if i.influential_citation_count is None else int(i.citation_count)

            # icite_importance = math.log10(icite_count + 1)+1
            # ref_importance = math.log(ref_time + 1)+1

            # importance = min(icite_importance*ref_importance,50)
            temp_IEI = get_IEI(i.title)['L6']
            temp_IEI = sigmoid(temp_IEI)

            temp_r = ((temp_IEI * 2) + 1) ** 2

            if temp_IEI < 0.5:
                print(get_IEI(i.title)['L6'], math.pi * (temp_r) ** 2)
            area = math.pi * ((temp_r) ** 2)

            start_year = pub_date.year
            start_month = pub_date.month
            end_year = sp_pub_date.year
            end_month = sp_pub_date.month

            diff_month = (end_year - start_year) * 12 + (end_month - start_month)

            times.append(diff_month)
            aFNCSIs.append(cur_aFNCSI)
            areas.append(area)

    x = np.array(times)
    y = np.array(aFNCSIs)
    colors = np.random.rand(x.shape[0])
    area = np.array(areas)

    plt.clf()

    plt.figure(figsize=(6, 4))
    # print(x.shape,y.shape,area.shape)
    colors = []
    for a in area:
        if a >= math.pi * (4) ** 2:
            colors.append(create_warm_colors())
        else:
            colors.append(create_cool_colors())

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.xlabel('Month Before Publication')

     
    plt.ylabel('aTNCSI')
    plt.savefig(f'{sp.title}.svg')


def _get_RQM(ref_obj, ref_type='entity', tncsi_rst=None):


    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False)

    loc = tncsi_rst['loc']
    scale = tncsi_rst['scale']




    # get Reference_relevant
    ref_r = s2paper.references
    N_R = len(ref_r)


    score = 0
    for item in ref_r:
        try:
            score += _get_TNCSI_score(item.citation_count, loc, scale)  # 1
        except:
            N_R = N_R - 1
            continue
    try:

        overlap_ratio = score / N_R # len(pub_set)
    except ZeroDivisionError:
        return 0

    # print(f'search paper title:{s2paper.title}, which has {len(ref_r)} refs. Due to errors, only count {N_R} papers.')
    return overlap_ratio

def get_RQM(ref_obj, ref_type='entity', tncsi_rst=None,beta=20):


    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False)

    loc = tncsi_rst['loc']
    scale = tncsi_rst['scale']


    pub_dates = []
    for i in s2paper.references:
        if i.publication_date:
            pub_dates.append(i.publication_date)
    # pub_dates = [i.publication_date for i in s2paper.references]
    sorted_dates = sorted(pub_dates, reverse=True)
    # index = 0
    # while index < len(sorted_list):
    #     try:
    #         sorted_list[index].timestamp()
    #         index += 1
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         del sorted_list[index]
     
    date_index = len(sorted_dates) // 2

     
    index_date = sorted_dates[date_index]

    # timestamps = [d.timestamp() for d in sorted_list]
    # if len(timestamps) == 0:
    #     return float('-inf')
    # median_timestamp = statistics.median(timestamps)
    # median_value = datetime.datetime.fromtimestamp(median_timestamp)
    # median_value = statistics.median(sorted_list)
    pub_time = s2paper.publication_date
    months_difference = (pub_time - index_date) // timedelta(days=30)
    S_mp = (months_difference // 6 ) + 1

    N_R = len(s2paper.references)


    score = 0
    for item in s2paper.references:
        try:
            score += _get_TNCSI_score(item.citation_count, loc, scale)  # 1
        except:
            N_R = N_R - 1
            continue
    try:
        ARQ = score / N_R # len(pub_set)
    except ZeroDivisionError:
        ARQ = 0
    rst = {}
    rst['RQM'] = 1 - math.exp(-beta * math.exp(-(1-ARQ) * S_mp))
    rst['ARQ'] = ARQ
    rst['S_mp'] = S_mp
    return rst

import numpy as np
import scipy.stats as stats
from scipy.integrate import cumtrapz
def get_RAD(M_pc):
    x = M_pc/12
    coefficients= np.array([-0.0025163,0.00106611, 0.12671325, 0.01288683])

    polynomial_function = np.poly1d(coefficients)
    x_pdf = np.linspace(0, 7.36, 200)  # extended range for PDF
    fitted_y_pdf = polynomial_function(x_pdf)
    pdf_normalized = fitted_y_pdf / np.trapz(fitted_y_pdf, x_pdf)


    # Compute the cumulative probability distribution function (CDF)
    cdf = cumtrapz(pdf_normalized, x_pdf, initial=0)
    cdf = np.where(cdf > 1.0, 1.0, cdf)
    # If x is less than the minimum x value in x_pdf, return 0 (assuming CDF is 0 at negative infinity)
    if x < x_pdf[0]:
        return 0

    # If x is greater than the maximum x value in x_pdf, return 1 (CDF is 1 at positive infinity)
    if x > x_pdf[-1]:
        return 1

    # Find the index in x_pdf that is closest to x
    index = np.searchsorted(x_pdf, x, side="left")

    # Return the corresponding CDF value
    return cdf[index]

def get_RUI(s2paper,p=10,q=10, M=None):
    """
    Calculate the integral of a log-normal distribution from 0 to t.

    :param mu: The mean (mu) of the log-normal distribution
    :param sigma: The standard deviation (sigma) of the log-normal distribution
    :param t: The upper limit of the integral
    :return: The integral of the log-normal distribution from 0 to t
    """
    # Calculate the cumulative distribution function (CDF) for log-normal distribution at t
    t = (datetime.now() - s2paper.publication_date) // timedelta(days=30)
    # RAD = stats.lognorm.cdf(t, s=sigma, scale=np.exp(mu))
    RAD = get_RAD(t)
    PC = request_query(s2paper.gpt_keyword,early_date=s2paper.publication_date)
    M = datetime.now() if not M else M
    MP = request_query(s2paper.gpt_keyword,early_date=get_median_pubdate(M,s2paper.references),later_date=s2paper.publication_date)
    rst = {}
    rst['RAD'] = RAD

    N_pc= PC['total']
    N_mp = MP['total']
    if N_mp == 0:
        return {'RAD':RAD, 'CDR':float('-inf'),'RUI':float('-inf')}

    CDR = N_pc/N_mp
    rst['CDR'] = CDR
    rst['RUI'] = p*CDR + q*RAD
    return rst
# s2paper = S2paper('Image segmentation using deep learning: A survey')
# rqm = get_RQM(s2paper, ref_type='entity')
# print(rqm)
