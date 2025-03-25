 
from arxiv import arxiv, SortCriterion, SortOrder
from jinja2 import Template

def simple_query_arxiv():
    rst = []
    query = f'all:"infrared" AND all:"small" AND all:"Target detection"'

    search = arxiv.Search(query=query,
                          max_results = float('inf'),
                          sort_by=SortCriterion.Relevance,
                          sort_order=SortOrder.Descending,
                          )

    for i,result in enumerate(search.results()):
        # print(result._get_default_filename)
        # if int(result.published.year)>=2022:
        # if check_relevant(result.title,result.summary,['CLIP','knowledge distillation']) == 'Y':
        print(i, result.title, result.entry_id, result.published, )
        print(result.summary)
        print('\n\n')
        rst.append(result)
    return

def simple_query_s2():
    rst = []
    query = f'all:"infrared" AND all:"small" AND all:"Target detection"'

    search = arxiv.Search(query=query,
                          max_results = float('inf'),
                          sort_by=SortCriterion.Relevance,
                          sort_order=SortOrder.Descending,
                          )

    for i,result in enumerate(search.results()):
        # print(result._get_default_filename)
        # if int(result.published.year)>=2022:
        # if check_relevant(result.title,result.summary,['CLIP','knowledge distillation']) == 'Y':
        print(i, result.title, result.entry_id, result.published, )
        print(result.summary)
        print('\n\n')
        rst.append(result)
    return







 
rst = simple_query()
papers = []
for p in rst:
    dct = {
        'title': p.title,
        'publish_date': p.published,
        'citation_count': 10,
        'content': p.summary,
    }
    # {
    #     'title': 'Paper 1',
    #     'publish_date': p,
    #     'citation_count': 10,
    #     'content': 'Content of Paper 1'
    # },
    papers.append(dct)


 
html_template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            padding: 20px;
        }
        .content {
            display: none;
            margin-top: 10px;
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }
        .paper-info {
            font-size: 14px;
            margin-bottom: 5px;
        }
        .publish-date {
            font-size: 12px;
            margin-right: 10px;
        }
        .citation-count {
            font-size: 12px;
        }
        .copy-button {
            font-size: 12px;
            background-color: #4caf50;
            color: #ffffff;
            border: none;
            border-radius: 3px;
            padding: 3px 6px;
            margin-left: 10px;
            cursor: pointer;
        }
        .hidden {
        display: none;
      }
    </style>
    <script>
        function toggleContent(index) {
            var content = document.getElementById('content-' + index);
            content.style.display = (content.style.display === 'none') ? 'block' : 'none';
        }

        function copyCitation(index) {
            var citation = document.getElementById('bibtex-' + index).textContent;
            var tempInput = document.createElement('input');
            tempInput.value = citation;
            document.body.appendChild(tempInput);
            tempInput.select();
            document.execCommand('copy');
            document.body.removeChild(tempInput);
            alert('Citation copied to clipboard!');
        }
    </script>
</head>
<body>
    {% for paper in papers %}
    <div>
        <h3 onclick="toggleContent({{ loop.index0 }})">{{ paper.title }}
            <button class="copy-button" onclick="copyCitation({{ loop.index0 }})">Copy Citation
            </button>
        </h3>
        <div class="paper-info">
            <span class="publish-date">Published: {{ paper.publish_date }}</span>
            <span class="citation-count">Citations: {{ paper.citation_count }}</span>
            <span id="bibtex-{{ loop.index0 }}" class="hidden">ccctestbibbibtex-{{ loop.index0 }}</span>
        </div>
        <div id="content-{{ loop.index0 }}" class="content">
            <p id="citation-{{ loop.index0 }}">{{ paper.content }}</p>
        </div>
    </div>
    {% endfor %}
</body>
</html>
"""

 
template = Template(html_template)
rendered_html = template.render(papers=papers)

 
with open('generated_page.html', 'w') as file:
    file.write(rendered_html)