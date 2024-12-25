# https://api.chatanywhere.cn
import os

from langchain_community.chat_models.openai import ChatOpenAI
from retry import retry

# openai.api_base = "https://api.chatanywhere.com.cn"
# openai.api_key = openai_key
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

def _get_ref_list(text):
    messages = [
        SystemMessage(content="You are a researcher, who is good at reading academic paper, and familiar with all of the "
                    "citation style. Please note that the provided citation text may not have the correct line breaks "
                    "or numbering identifiers."),
        HumanMessage(content=f'''
        Extract the paper title only from the given reference text, and answer with the following format.
                [1] xxx
                [2] xxx
                [3] xxx 
            Reference text: {text}
        '''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content


@retry(delay=6,)
def get_chatgpt_keyword(title, abstract):
    messages = [
        SystemMessage(content="You are a profound researcher in the field of artificial intelligence who is good at selecting "
                    "keywords for the paper with given title and abstract. Here are some guidelines for selecting keywords: 1. Represent the content of the title and abstract. 2. Be specific to the field or sub-field. "
                    "3. Keywords should be descriptive. 4. Keywords should reflect a collective understanding of the topic. 5. If the research paper involves a key method or technique, put the term in keywords"),
        HumanMessage(content=f'''Summarize 3-5 keywords only from the given title and abstract, and answer with the following format: xxx, xxx, ..., xxx,
            Given Title: {title}
            Given Abstract: {abstract}
'''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content




def get_unnum_sectitle(sectitle):

    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of artificial intelligence who have read a lot of "
                    "paper. You can figure out what is the title of section, irrespective of whether they are "
                    "numbered or unnumbered, and the specific numbering format utilized."),
        HumanMessage(content=f'This is the title of section, extract the title without chapter numbering(If chapter numbering exists). Answer with the following format: xxx. \n Section Title: {sectitle}'),
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content
@retry()
def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True,model="gpt-3.5-turbo-0125",temperature=0):

    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic key phrase from paper's title and "
            "abstract. Ensure that the topic key phrase precisely defines the research area within the article. For effective academic searching, such as on Google Scholar, the field should be specifically targeted rather than broadly categorized. For instance, use 'image classification' instead of the general 'computer vision' to enhance relevance and searchability of related literature.")
    if not usr_prompt:
        usr_prompt = ("Given the title and abstract below, determine the specific research field by focusing on the main application area and the key technology. You MUST respond with the keyword ONLY in this format: xxx")

    messages = [SystemMessage(content=sys_content)]

    extra_abs_content = '''
    Given Title: Large Selective Kernel Network for Remote Sensing Object Detection
    Given Abstract: Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, which can vary for different objects. This paper considers these priors and proposes the lightweight Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To our knowledge, large and selective kernel mechanisms have not been previously explored in remote sensing object detection. Without bells and whistles, our lightweight LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.85% mAP), and FAIR1M-v1.0 (47.87% mAP).''' if abstract else ''
    if extra_prompt:
        messages += [HumanMessage(content=f'''{usr_prompt}\n\n{extra_abs_content}'''), AIMessage(content='remote sensing object detection')]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'
    messages.append(HumanMessage(content=content))

    chat = ChatOpenAI(model=model,temperature=temperature)



    return chat.batch([messages])[0].content
@retry()
def get_chatgpt_fields(title, abstract, extra_prompt=True,sys_content=None,usr_prompt=None):

    if not sys_content:
        sys_content = ("You are a profound researcher who is good at conduct a literature review based on given title and abstract.")
    if not usr_prompt:
        usr_prompt = ("Given title and abstract, please provide 5 seaching keywords for me so that I can use them as "
                      "keywords to search highly related papers from Google Scholar or Semantic Scholar. Please avoid "
                      "responding with overly general keywords such as deep learning, taxonomy, or surveys, etc., "
                      "and provide the output in descending order of relevance to the keywords. Answer with the words "
                      "only in the following format: xxx,xxx,xxx")

    if extra_prompt:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n Given Title: Diffusion Models in Vision: A Survey \nGiven Abstract: Denoising 
             diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results 
             in the area of generative modeling. A diffusion model is a deep generative model that is based on two 
             stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, 
             the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, 
             a model is tasked at recovering the original input data by learning to gradually reverse the diffusion 
             process, step by step. Diffusion models are widely appreciated for the quality and diversity of the 
             generated samples, despite their known computational burdens, i.e., low speeds due to the high number of 
             steps involved during sampling. In this survey, we provide a comprehensive review of articles on 
             denoising diffusion models applied in vision, comprising both theoretical and practical contributions in 
             the field. First, we identify and present three generic diffusion modeling frameworks, which are based 
             on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic 
             differential equations. We further discuss the relations between diffusion models and other deep 
             generative models, including variational auto-encoders, generative adversarial networks, energy-based 
             models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective 
             categorization of diffusion models applied in computer vision. Finally, we illustrate the current 
             limitations of diffusion models and envision some interesting directions for future research.'''),
                    AIMessage(content='Denoising diffusion models,deep generative modeling,diffusion models,image generation,noise conditioned score networks'),
                    HumanMessage(content=f'''{usr_prompt}\n
                            Given Title: {title}\n
                            Given Abstract: {abstract}
                        ''')]
    else:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n
                Given Title: {title}\n
                Given Abstract: {abstract}
            ''')]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content



@retry(delay=6, )
def extract_keywords_from_article_with_gpt(text):
    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of pattern recognition and machine intelligence. You "
                    "are aware of all types of keywords, such as keyword, index terms, etc.Please note: The text is "
                    "extracted from the PDF, so line breaks may appear anywhere, or even footnotes may appear between "
                    "consecutive lines of text."),
        HumanMessage(content= f'''I will give you the text in the first page of an academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page: Cryoelectron Microscopy as a Functional Instrument for Systems Biology, Structural Analysis &
        Experimental Manipulations with Living Cells
        (A comprehensive review of the current works).
        Oleg V. Gradov
        INEPCP RAS, Moscow, Russia
        Email: o.v.gradov@gmail.com
        Margaret A. Gradova
        ICP RAS, Moscow, Russia
        Email: m.a.gradova@gmail.com
        Abstract — The aim of this paper is to give an introductory
        review of the cryoelectron microscopy as a complex data source
        for the most of the system biology branches, including the most
        perspective non-local approaches known as "localomics" and
        "dynamomics". A brief summary of various cryoelectron microscopy methods and corresponding system biological approaches is given in the text. The above classification can be
        considered as a useful framework for the primary comprehensions about cryoelectron microscopy aims and instrumental
        tools
        Index Terms — cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines
        I. TECHNICAL APPLICATIONS OF
        CRYOELECTRON MICROSCOPY
        Since its development in early 1980s [31]
        cryo-electron microscopy has become one of
        the most functional research methods providing
        the study of physiological and biochemical
        changes in living matter at various hierarchical
        levels from single mammalian cell morphology
        [108] to nanostructures
    '''),
        AIMessage(content=f'''cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines'''),
        HumanMessage(content=f'''I will give you the text in the first page of another academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page:{text}''')
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content

