from datetime import datetime
import os
import re
import shelve

import requests
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from CACHE.CACHE_Config import generate_cache_file_name
from database.DBEntity import AuthorMapping, AoP, PaperMapping, RefMapping, RoP, CoP
from furnace.Author import Author
from furnace.arxiv_paper import Arxiv_paper

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from furnace.google_scholar_paper import Google_paper
from furnace.semantic_scholar_paper import S2paper
from tools.gpt_util import *

Base = declarative_base()

ref_cache = r'../CACHE/.refCache'

 
engine = create_engine('xxxliteraturedatabase')

 
Base.metadata.create_all(engine)

 
Session = sessionmaker(bind=engine)
session = Session()


def add_info_to_database(dir: str, session):
    '''
    从arxiv下载后论文的第一步
    :param dir:
    :param session:
    :return:
    '''
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            title_parts = filename.split('.')
            if match:
                # arxiv_id = match.group()
                arxiv_id = '.'.join(title_parts[:2])
                cur_paper = Arxiv_paper(arxiv_id, ref_type='id')

                id = 'http://arxiv.org/abs/' + arxiv_id
                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                # print(data.idLiterature)
                if data is None:
                    print(arxiv_id)
                    _ = cur_paper.entity

                    doc = PaperMapping(arxiv_paper=cur_paper)
                     
                     
                    session.add(doc)
                     
                    session.commit()
                else:
                    print('done')

            else:
                print("No arXiv ID found.")

     
    session.close()


# add_info_to_database(r'D:\download_paper',session)

import time
import random


def update_keyword(dir: str, session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            # print(filename)
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)

            title_parts = filename.split('.')
            if match:
                arxiv_id = '.'.join(title_parts[:2])
                id = 'http://arxiv.org/abs/' + arxiv_id

                 

                 
                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                # print(data.idLiterature)
                if data and data.gpt_keywords is None:
                    time.sleep(random.uniform(5.5, 8))
                    try:
                        kwd = get_chatgpt_keyword(data.title, data.abstract.replace('\n', ''))
                    except Exception as e:
                        print(e)
                        print(data.title)
                        print(data.abstract)
                        return
                     
                    # kwd=''
                    data.gpt_keywords = ','.join(kwd)
                     
                    # print(','.join(kwd))
                    try:
                         
                        if data.citation_count is None:
                            g_paper = Google_paper(data.title)

                            if g_paper.citation_count >= 0:
                                data.citation_count = g_paper.citation_count
                                # print(data.citation_count)
                            if g_paper.publication_source != 'NA':
                                data.publication_source = g_paper.publication_source
                                # print(data.publication_source)

                    except KeyError as E:
                        print(E)
                        return
                    #
                    session.commit()
                     
                else:
                    pass
                     
                 
                # for row in data:

             
            session.close()

     
    session.close()


import json


def update_author(session):
    '''
    更新作者信息 不要轻易使用，最后一步使用
    :param session:
    :return:
    '''
     
    results = session.query(PaperMapping).all()
    for row in results:

        authors_str = row.authors.replace("} {", "}, {")
         
        author_lst = []
        for item in json.loads("[" + authors_str + "]"):
            name = item.get("name")
            if name:
                am = AuthorMapping(Author(name))

                author_lst.append(am)
        session.bulk_save_objects(author_lst)
        aop = AoP(author_lst)
        session.add(aop)

        row.aop = aop.idaop

    session.commit()

    # author_mapping = []
    # for


# update_keyword(r'D:\download_paper',session)
def get_keywords():
    '''
    更逊作者信息 不要轻易使用
    :param session:
    :return:
    '''
     
    results = session.query(PaperMapping).all()
    rst = []
    for row in tqdm(results):
        rst += row.gpt_keywords.split(',')
    session.close()
    return rst


def update_s2(session, mandatory_update=False):
    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        if (row.s2_id is None or mandatory_update) and (row.valid==1 and row.is_review==1):
            # if row.authors_num is not None:
            #     continue
            s2_paper = S2paper(row.title, filled_authors=False, force_return=True)
            if s2_paper.s2id is not None:

                if s2_paper.title != row.title:
                    if s2_paper.authors[0].name in row.authors:
                        print(f'The same paper with different titles detected: {row.title} \n {s2_paper.title}')
                    else:
                        continue
                time.sleep(0.5)
                row.s2_id = s2_paper.s2id
                row.s2_publication_date = s2_paper.publication_date
                row.s2_tldr = s2_paper.tldr
                row.s2_DOI = s2_paper.DOI
                row.s2_pub_info = s2_paper.publication_source
                row.s2_pub_url = s2_paper.pub_url['url'][:255] if s2_paper.pub_url else None
                row.s2_citation_count = s2_paper.citation_count
                row.s2_reference_count = s2_paper.reference_count
                row.s2_field = s2_paper.field
                row.s2_influential_citation_count = s2_paper.influential_citation_count
                row.valid = 1
                row.last_update_time = datetime.now()
                # print(s2_paper.authors)
                # row.authors = s2_paper.authors
                # print(s2_paper)
                row.authors_num = len(s2_paper.authors) if s2_paper.authors is not None else None
            else:
                row.valid = 0
            session.commit()
    session.close()


# update_s2(session)
import json
from config.config import *
from sqlalchemy import and_
from retry import retry
from sqlalchemy.sql import text


@retry()
def update_s2_ref(session):
    # metadata = MetaData(bind=engine)
    #
    # table = Table('rop', metadata, autoload=True)
    # delete_stmt = table.delete()
    # session.execute(delete_stmt)
    # session.commit()
    #
    # table = Table('reference', metadata, autoload=True)
    # delete_stmt = table.delete()
    # session.execute(delete_stmt)
    # session.commit()

    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        # if row.title == 'malaria likelihood prediction by effectively surveying households using deep reinforcement learning':
        #     pass
        if row.is_review == 1 and row.valid == 1:# row.reference_details is None and

            url = f'https://api.semanticscholar.org/graph/v1/paper/{row.s2_id}/references?fields=paperId,title,authors,intents,contexts,isInfluential,url,publicationVenue,openAccessPdf,abstract,venue,year,referenceCount,citationCount,influentialCitationCount,s2FieldsOfStudy,publicationTypes,journal,publicationDate&offset=0&limit=1000'

            with shelve.open(generate_cache_file_name(url)) as cache:
                # ref_count = row.s2_reference_count

                if url in cache:
                    # print('Cache loding')
                    reply = cache[url]
                    # print('Cache Loading Done')
                else:
                    # continue
                    req_session = requests.Session()
                    # s2api = None
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    reply = req_session.get(url, headers=headers)
                    cache[url] = reply

                    time.sleep(0.33)
            try:
                response = reply.json()  # 0.15s
            except:
                continue
            # print('ssadad')

            if "data" not in response:
                row.valid = -1

            else:
                row.reference_details = json.dumps(response) #str(response['data'])
                session.commit()
    session.close()
    # ref_list = []
    # rop_list = []
    # for entity in response['data']:
    #     contexts = entity['contexts']
    #     intents = entity['intents']
    #     isInfluential = entity['isInfluential']
    #     # print(isInfluential,type(isInfluential))
    #     entity = entity['citedPaper']
    #     # start_time = time.time()
    #     s2paper = S2paper(entity,filled_authors=False,ref_type='entity')
     
    #
    #     # check_exist = session.query(RefMapping).filter(RefMapping.s2_id == s2paper.s2id).first()
    #     # end_time = time.time()
    #     # execution_time = end_time - start_time
     
    #     #
    #     # print(data.idLiterature)
    #
    #     query = text("SELECT * FROM reference WHERE s2_id = :s2_id LIMIT 1")
    #     result = session.execute(query, {"s2_id": s2paper.s2id})
    #     check_exist = result.fetchone()
    #
    #
    #     # session.commit()
    #     if check_exist is None:
    #         ref_data = RefMapping(s2paper, contexts, intents, isInfluential)
    #         ref_list.append(ref_data)
    #     # session.add(ref_data)
    #     # session.commit()
    #     else:
    #         # print('Ref Exist')
    #         ref_data = check_exist
    #
    #
    #     # check_exist = session.query(RoP).filter(and_(RoP.idLiterature == row.idLiterature, RoP.idReference == ref_data.idReference)).first()
    #
    #
    #     query = text("SELECT * FROM rop WHERE idLiterature = :literature_id AND idReference = :reference_id LIMIT 1")
    #     result = session.execute(query,
    #                              {"literature_id": row.idLiterature, "reference_id": ref_data.idReference})
    #     check_exist = result.fetchone()
    #     if check_exist is None:
    #
    #         rop_data = RoP(row.idLiterature, ref_data.idReference)
    #         rop_list.append(rop_data)
    #     else:
    #         # print('RoP Exist')
    #         pass
     

    # session.add(rop_data)
    # session.commit()
    # time.sleep(0.1)
    # print(s2_paper.title)
    # print(len(ref_list))
    # print(len(rop_list))
    # session.flush()

    # session.bulk_save_objects(ref_list)
    # # time.sleep(1)
    # session.bulk_save_objects(rop_list)

    # session.expire_all()

    # session.flush()


@retry()
def update_s2_citation(session):
    results = session.query(CoP).all()

    for row in tqdm(results):
        if row.full_citation is None:
            OFFSET = 0
            citation_responses = []
            if True: #row.citation is None:
                while (True):
                    url = f'https://api.semanticscholar.org/graph/v1/paper/{row.s2_id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset={OFFSET}&limit=1000'

                    with shelve.open(generate_cache_file_name(url)) as cache:
                        print(url)
                        if url in cache:
                            # print('Cache loding')
                            reply = cache[url]
                            # print('Cache Loading Done')
                        else:
                            time.sleep(1)
                            req_session = requests.Session()
                            # s2api = None
                            if s2api is not None:
                                headers = {
                                    'x-api-key': s2api
                                }
                            else:
                                headers = None
                            reply = req_session.get(url, headers=headers,verify=False)

                            cache[url] = reply

                        try:
                            response = reply.json()  # 0.15s
                        except:
                            OFFSET += 1000
                            continue

                        if "data" in response:
                            citation_responses.append(response)
                            if response['data'] == []:
                                break
                            OFFSET += 1000
                            # cache[url] = reply

                        else:
                            break


                        #

                row.full_citation = json.dumps(citation_responses)
                session.commit()

    session.close()


def insert_idLiterature_into_CoP(session):
    metadata = MetaData(bind=engine)
    table = Table('cop', metadata, autoload=True)
    delete_stmt = table.delete()
    session.execute(delete_stmt)
    session.commit()

    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        if row.valid==1 and row.is_review ==1:
            try:
                cop_entry = CoP(row.s2_id, citation=None)
                session.add(cop_entry)
                session.commit()
            except SQLAlchemyError as e:
                 

                session.rollback()
                row.valid = -1
                session.commit()
                print(e)
                # continue




# gpt_process(session)
# update_s2_ref(session)


def sync_folder_to_database(session, dir=None):
     
    # add_info_to_database(dir, session)
     
    # gpt_process(session)

     
    print('查询s2信息')
    # update_s2(session, True)
    print('更新cite,ref信息')
     
    # update_s2_ref(session)
    update_s2_citation(session)

     
    # update_author(session)


def update_official_keywords(dir: str, session):
    import PyPDF2
    def extract_text_from_pdf(pdf_file_path):
        try:
             
            with open(pdf_file_path, 'rb') as pdf_file:
                 
                pdf_reader = PyPDF2.PdfFileReader(pdf_file)

                 
                if pdf_reader.numPages > 0:
                     
                    page = pdf_reader.getPage(0)

                     
                    text = page.extractText()

                    return text
                else:
                    return ''
        except Exception as e:
            return ''

    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            title_parts = filename.split('.')
            if match:

                # arxiv_id = match.group()
                arxiv_id = '.'.join(title_parts[:2])

                id = 'http://arxiv.org/abs/' + arxiv_id
                try:
                    data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                except Exception as e:
                    print(e)
                    continue
                # print(data.idLiterature)
                if data:
                    if data.keywords is None:

                        # keywords = [keyword.replace('.', '').replace("'", "").replace('"', "") for keyword in keywords]
                        #
                        # keywords = keywords[:5]
                        # # row.gpt_keywords = ','.join(keywords)
                        # row.gpt_keywords = ','.join(keywords)
                        #
                        # session.commit()
                        extracted_text = extract_text_from_pdf(os.path.join(dir, filename))
                        keywords = extract_keywords_from_article_with_gpt(extracted_text)
                        keywords = [keyword.replace('.', '').replace("'", "").replace('"', "") for keyword in keywords]
                        data.keywords = ';'.join(keywords)

                        if len(data.keywords) >= 255:
                            data.keywords = 'None'
                        try:
                            # print(data.keywords)
                            session.commit()
                        except Exception as e:
                            print(e)
                            continue

            else:
                print("No arXiv ID found.")

     
    session.close()

def update_gpt_keyword(session):
    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        # print(row.title)
        if row.gpt_keyword is None and row.is_review == 1 and row.valid == 1:
            keywords = get_chatgpt_field(row.title, row.abstract)
            keywords = [keyword.replace('.', '').replace("'", "").replace('"', "") for keyword in keywords]

            keyword = keywords[0]
            # row.gpt_keywords = ','.join(keywords)
            row.gpt_keyword = str(keyword)

            session.commit()
        # else:
        #     row.gpt_keywords = row.gpt_keywords.replace('.', '').replace("'", "").replace('"', "")
        #     session.commit()
        # if row.is_review is None:
        #     status = check_PAMIreview(row.title, row.abstract)
        #     if status == ('N' or 'n'):
        #         row.is_review = 0
        #     if status == ('Y' or 'y'):
        #         row.is_review = 1
        #     session.commit()
        # print(f'{row.title}||{row.is_review}||{row.gpt_keywords}')
    session.close()

if __name__ == "__main__":
    # update_gpt_keyword(session)
    sync_folder_to_database(session,dir=r'E:\download_paper')
    # update_official_keywords(r'E:/download_paper', session)
    # insert_idLiterature_into_CoP(session)
    # update_s2_citation(session)
    pass
