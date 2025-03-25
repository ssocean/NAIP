import logging
import pickle
import random
import shelve
import string
import time
import warnings

from deprecated import deprecated

from requests.exceptions import SSLError
from retry import retry
from scipy.stats import spearmanr, pearsonr

from CACHE.CACHE_Config import generate_cache_file_name
from config.config import eskey, openai_key, s2api
from retriever.semantic_scholar_paper import S2paper, request_query
from tools.Cache import make_cached_request, dump_cache
from tools.Reference import Ref

import openai

from tools.gpt_util import _get_ref_list


openai.api_key = openai_key
import math
import re
from pdfminer.high_level import extract_text
from scholarly import scholarly
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PE = 'xxx'  # polite Email


def get_avg_cite_num(query, year_low: int = None, year_high: int = None, topk=100, remove_k=None):
    citation_count = []
    if isinstance(query, str):

        search_query = scholarly.search_pubs(query, patents=False, year_low=year_low, year_high=year_high)
        '''
        patents: bool = True,
                        citations: bool = True, year_low: int = None,
                        year_high: int = None, sort_by: str = "relevance",
                        include_last_year: str = "abstracts",
                        start_index: int = 0
        '''
         

        for i, pub in tqdm(enumerate(search_query)):
            print(pub['bib']['title'])
            if i < topk:
                citation_count.append(pub['num_citations'])
            else:
                break
        # citation_count = citation_count[:topk]


    elif isinstance(query, list):
        assert len(query) > 0 and isinstance(query[0], Ref), 'query eror'
        for ref in tqdm(query):
            citation_count.append(ref.citations)

    else:
        raise NotImplementedError
    if remove_k:
        sorted(citation_count)
        citation_count = citation_count[remove_k:len(citation_count) - remove_k]
    return int(sum(citation_count) / len(citation_count)) if len(citation_count) > 0 else 0


def get_cite_score(cur_cite, avg_cite):
    assert cur_cite >= 0, 'The num of citations should larger than zero!'
    if cur_cite == 0:
        return 0.5
    relative_cite = cur_cite - avg_cite
    if relative_cite >= 0:
        return (1 / (1 + math.exp(-(1 / avg_cite) * relative_cite))) + 1
    else:
        # relative_cite = -relative_cite
        return (math.log(cur_cite + 1, avg_cite + 1) / 2) + 1


import requests


def get_cited_by(search_pub):
    return scholarly.bibtex(search_pub)


def extract_ref(pdf_pth):
    ref_keywords = ['Reference', 'Bibliograph', 'List of Reference', 'Reference List']
     
    with open(pdf_pth, "rb") as f:
         
        text = extract_text(f)
         
        for ref_keyword in ref_keywords:
            start = re.search(ref_keyword, text)
            if start:
                content = text[start.start():]
                return content
            else:
                print(f"Chapter not found. Searching for new one: {ref_keyword}")
    raise EOFError


def get_ref_list(text: str, span=3500):
    words = text.split(' ')
    if len(words) < span:
        result = _get_ref_list(text)
    else:
        start_index = 0
        end_index = span
        result = []
        while end_index < len(words):
            ref_text = ' '.join(words[start_index:end_index])
            result += _get_ref_list(ref_text)
            start_index += span - 200
            end_index = start_index + span
    return list(set(result))


def cal_cocite_rate(ref_list_Anchor, ref_list_eval):
    ref_list_Anchor_titles = [i.title.lower() for i in ref_list_Anchor]
    ref_list_eval_titles = [i.title.lower() for i in ref_list_eval]

    anchor_set = set(ref_list_Anchor_titles)
    eval_set = set(ref_list_eval_titles)

    intersection = anchor_set & eval_set
    try:
        overlap_ratio = len(intersection) / len(anchor_set)
    except ZeroDivisionError:
        return 0
    return overlap_ratio


def cal_weighted_cocite_rate(ref_list_Anchor, ref_list_eval, avg_cite_num):
    ref_list_Anchor_titles = [(i.title.lower(), i.citations) for i in ref_list_Anchor]
    ref_list_eval_titles = [(i.title.lower(), i.citations) for i in ref_list_eval]

    anchor_set = set(ref_list_Anchor_titles)
    eval_set = set(ref_list_eval_titles)

    intersection = anchor_set & eval_set
    intersection = list(intersection)
    score = 0
    for item in intersection:
        score += (get_cite_score(item[-1], avg_cite_num) + 0.5)
    try:
        overlap_ratio = score / len(anchor_set)
    except ZeroDivisionError:
        return 0
    return overlap_ratio


# a = eval_writting_skill('LitStudy is a Python package that enables analysis of scientific literature from the comfort of a Jupyter notebook. It provides the ability to select scientific publications and study their metadata through the use of visualizations, network analysis, and natural language processing.')
# print(a)
# a = eval_writting_skill('LitStudy is a Python package that enable analysis of scientific literatures from the comfort of Jupyter notebook. It provide the ability to select scientific publications and study their metadata through the use of visualizations, network analysis, and natural language processing.')
# print(a)
# b = eval_writting_skill("Machine learning are typically framed from a perspective of i.i.d., and more importantly, isolated data. In parts, federated learning lifts this assumption, as it sets out to solve the real-world challenge of collaboratively learning a shared model from data distributed across clients. However, motivated primarily by privacy and computational constraints, the fact that data may change, distributions drift, or even tasks advance individually on clients, is seldom taken into account. The field of continual learning addresses this separate challenge and first steps have recently been taken to leverage synergies in distributed supervised settings, in which several clients learn to solve changing classification tasks over time without forgetting previously seen ones. Motivated by these prior works, we posit that such federated continual learning should be grounded in unsupervised learning of representations that are shared across clients; in the loose spirit of how humans can indirectly leverage others' experience without exposure to a specific task. For this purpose, we demonstrate that masked autoencoders for distribution estimation are particularly amenable to this setup. Specifically, their masking strategy can be seamlessly integrated with task attention mechanisms to enable selective knowledge transfer between clients. We empirically corroborate the latter statement through several continual federated scenarios on both image and binary datasets.")
# print(b)

import datetime

import arxiv
from arxiv import SortCriterion, SortOrder

from tools.Reference import Ref


def get_arxiv(keywords, max_results=float('inf')):
    if ':' in keywords:
        query = keywords
    else:
        query = f'abs:"{keywords.lower()}"'

    search = arxiv.Search(query=query,
                          max_results=max_results,
                          sort_by=SortCriterion.Relevance,
                          sort_order=SortOrder.Descending,
                          )
    return search


def filter_arxiv(search, filter_keys='',
                 start_time=datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                 end_time=datetime.datetime.now(tz=datetime.timezone.utc), max_results=-1):
    print("all search:")
    for index, result in enumerate(search.results()):
        print(index, result.title, result.updated)

    filter_results = []

    print("filter_keys:", filter_keys)
     
    for index, result in enumerate(search.results()):
        abs_text = result.summary.replace('-\n', '-').replace('\n', ' ')
        meet_num = 0
        for f_key in filter_keys.split(" "):
            if f_key.lower() in abs_text.lower():
                meet_num += 1
        if meet_num == len(filter_keys.split(" ")):
            filter_results.append(result)
            # break

    rst = []
    for i, result in enumerate(filter_results):
        if result.published < end_time and result.published > start_time:
            rst.append(result)
    print("筛选后剩下的论文数量：")
    print("filter_results:", len(rst))
    print("filter_papers:")
    for index, result in enumerate(rst):
        print(index, result.title, result.updated)
    if max_results > 0:
        return rst[:max_results]
    return rst


def arxiv2ref(arxiv_search):
    rst = []
    for item in arxiv_search:
        rst.append(Ref(item.title, extract_title=False))
    return rst


#
from scholarly import scholarly
from tqdm import tqdm


def get_google(query, topk=100, year_low: int = None,
               year_high: int = None, sort_by: str = "relevance"):
    search_query = scholarly.search_pubs(query, patents=False, year_low=year_low, year_high=year_high, sort_by=sort_by)
    rst = []
    for i, pub in tqdm(enumerate(search_query)):
        if i < topk:
            rst.append(Ref('', pub_obj=pub, search_from='google'))
        else:
            return rst
    '''
    patents: bool = True,
                    citations: bool = True, year_low: int = None,
                    year_high: int = None, sort_by: str = "relevance",
                    include_last_year: str = "abstracts",
                    start_index: int = 0
    pass'''


import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_next_month(year_month: str):
     
    year, month = map(int, year_month.split('.'))
    date_obj = datetime(year, month, 1)
    next_month = date_obj + relativedelta(months=1)
    next_month_str = f'{next_month.year}.{next_month.month}'

    return next_month_str


def get_year_month_list(start_year_month, end_year_month=None):
    lst = []
    cur_year_month = start_year_month
    lst.append(cur_year_month)
    if end_year_month is None:
        end_year_month = f'{datetime.datetime.now().year}.{datetime.datetime.now().month}'
    while cur_year_month != end_year_month:
        print(cur_year_month, end_year_month)
        cur_year_month = get_next_month(cur_year_month)
        lst.append(cur_year_month)
    return lst


import pandas as pd
import matplotlib.pyplot as plt


@retry()
def safe_retry_cited_by(cited_by, tqdm_total):
    time_freq = {}
    for index, i in enumerate(tqdm(cited_by, total=tqdm_total)):

        # print(info['message']['items'][0])
        cited_by_ref = Ref(i, ref_type='google')
        print(cited_by_ref.title)
        if cited_by_ref.pub_date:
            time_freq.update({cited_by_ref.pub_date: time_freq.get(cited_by_ref.pub_date, 0) + 1})
        #         time_freq.update({cited_by_ref.pub_date: time_freq.get(cited_by_ref.pub_date, 0) + 1})
        else:
            return None
        time.sleep(random.uniform(0.0, 0.34))
    return time_freq


def pub_number_histogram(ref_obj, output_pth=None):
    # = {}
    search_cite_paper = Ref(ref_obj)
    if not search_cite_paper.cited_by:
        warnings.warn("Can not fetch cited_by", UserWarning)
        return None

    time_freq = safe_retry_cited_by(search_cite_paper.cited_by, tqdm_total=search_cite_paper.citations)
    # for index, i in enumerate(tqdm(search_cite_paper.cited_by, total=search_cite_paper.citations)):
    #     # print(info['message']['items'][0])
    #     cited_by_ref = Ref(i, ref_type='google')
    #     print(cited_by_ref.title)
    #     if cited_by_ref.pub_date:
    #         time_freq.update({cited_by_ref.pub_date: time_freq.get(cited_by_ref.pub_date, 0) + 1})
    #     #         time_freq.update({cited_by_ref.pub_date: time_freq.get(cited_by_ref.pub_date, 0) + 1})
    #     else:
    #         return None
    #     time.sleep(random.uniform(0.0, 0.34))
    # if 'arxiv' in cited_by_ref.venue.lower():
    #
    #     time_freq.update({cited_by_ref.pub_date: time_freq.get(cited_by_ref.pub_date, 0) + 1})
    # else:
    #     cited_by_ref.enable_crossref()
    #     if cited_by_ref.crossref_rst:
    #         cited_by_ref.disable_arxiv()
    #         # tim = info['message']['items'][0]['created']['date-parts'][0]

    # search_title = info['message']['items'][0]['title'][0]

    # datetime(tim[0], tim[1], tim[2], 0, 0, 0)
    # dict_key = str(tim[0]) + '.' + str(tim[1])
    # time_freq.update({dict_key: time_freq.get(dict_key, 0) + 1})

    if output_pth:
         
        search_pub_date = search_cite_paper.pub_date
        dates = get_year_month_list(search_pub_date)
        pub_nums = []
        for date in dates:
            pub_num = time_freq[date] if date in time_freq else 0
            pub_nums.append(int(pub_num))
        data = {'Year': dates,
                'Pub num': pub_nums}
        df = pd.DataFrame(data)

         
        df.plot(x='Year', y='Pub num', kind='bar')
        plt.show()
    return time_freq


def plot_s2citaions(keyword: str, total_num=2000):
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
    # response = request_query(keyword,sort_rule,)
    sort_rule ='paperId:desc'
    # sort_rule = 'citationCount:desc'
    continue_token = None
    for i in range(0, total_num, 1000):
        if continue_token is None:
            response = request_query(keyword, sort_rule=sort_rule, pub_date=datetime.now())
        else:
            response = request_query(keyword, sort_rule=sort_rule, continue_token=continue_token,pub_date=datetime.now())

        if "token" in response:
            continue_token = response['token']
        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            logger.warning('No matched paper!')
            raise FileNotFoundError
            # raise Exception(f"error while fetching {reply.url}: {msg}")
        else:
            # for entity in response['data']:
            #     temp_ref = S2paper(entity, ref_type='entity', force_return=False, filled_authors=False)
            #     pub_r.append(temp_ref)
            for item in response['data'][:total_num]:
                if int(item['citationCount']) >= 0:
                    citation_count.append(int(item['citationCount']))
                    influentionCC.append(int(item['influentialCitationCount']))
                    l += 1
                else:
                    print(item['citationCount'])

        logger.info(f'Fetch {l} data from SemanticScholar.')
    return citation_count, influentionCC



@retry(tries=3)
def _plot_s2citaions(keyword: str, year: str = None, total_num=2000, CACHE_FILE='.ppicache'):
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
    with shelve.open(CACHE_FILE) as cache:
        for i in range(int(total_num / 100)):
            if year:
                url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&year={year}&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'
            else:
                url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'

            if url in cache:
                r = cache[url]
            else:

                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                r = requests.get(url, headers=headers,verify=False)
                time.sleep(0.5)
            try:
                if 'data' in r.json():
                    cache[url] = r
            # print(r.json())


                if 'data' not in r.json():
                    logger.info(f'Fetching {l} data from SemanticScholar.')
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


import numpy as np
from PIL import Image


def fig2img(fig):
    '''
    matplotlib.figure.Figure转为PIL image
    '''
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
     
    img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
    return img



# plot_s2citaions('segmentation',total_num=200)
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

@retry(tries=3)
def get_esPubRank(publication_name):
    import requests

    url = 'https://www.easyscholar.cc/open/getPublicationRank'
    secret_key = eskey



    params = {'secretKey': secret_key, 'publicationName': publication_name}
    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])

    # The complete URL
    url = f"{url}?{query_string}"
    # print(url)
    with shelve.open('.esPubrank') as cache:
        if url in cache:
            response = cache[url]
        else:
            response = requests.get(url,verify=False)
            time.sleep(0.5)
            cache[url] = response

    if response.status_code == 200:
        data = response.json()
        if data.get('data'):
            if data.get('data').get('officialRank'):
                if data.get('data').get('officialRank').get('all'):
                    return data['data']['officialRank']['all']

    return None


# print(get_esPubRank('IEEE Conference on Computer Vision and Pattern Recognition'))

from datetime import datetime, timedelta
from collections import OrderedDict


@retry()
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
                r = requests.get(url, headers=headers,verify=False)
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


def stat_citation(citaitons_dict, observed_duration=6):
    rst = {}
    data = []
    for spm in citaitons_dict.items():
        data.append(spm[1])
    data = data[1:1 + observed_duration]   
    data.reverse()
    # print(data)
    data = np.array(data)
    # print(data)
    month = [i for i in range(1, 1 + observed_duration)]
    month = np.array(month)

     
    spearman_correlation, _ = spearmanr(list(month), list(data))
    pearson_correlation, _ = pearsonr(list(month), list(data))

     
    rst['spearman_correlation'] = spearman_correlation
    rst['pearson_correlation'] = pearson_correlation
     

     
    diff_matrix = np.subtract.outer(data, data)
    corr_matrix = np.corrcoef(month, data)
     
    # print(-diff_matrix)
    # print(corr_matrix)

    diff_matrix = -diff_matrix
     
    upper_diagonal_elements = diff_matrix[np.triu_indices(diff_matrix.shape[0], k=1)]

     
    dm_mean = np.mean(upper_diagonal_elements)
    dm_variance = np.var(upper_diagonal_elements)
    rst['dm_mean'] = dm_mean
    rst['dm_variance'] = dm_variance

    mean = np.mean(data)
    var = np.var(data)
    rst['mean'] = mean
    rst['variance'] = var
    return rst


if __name__ == "__main__":
    c_dict = get_s2citaions_per_month('Learning to Prompt for Vision-Language Models', 10000)
    # print(c_dict)
    stat = stat_citation(c_dict)
    print(stat)
