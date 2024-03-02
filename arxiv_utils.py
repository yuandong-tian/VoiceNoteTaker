import requests
import os
import re
from bs4 import BeautifulSoup
import tarfile

def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """

    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False

def expand_inputs(base_path, file_content):
    """
    Recursively expands \input and \include commands in a LaTeX file content.
    
    :param base_path: The base directory of the LaTeX files.
    :param file_content: The content of the LaTeX file to process.
    :return: The expanded LaTeX content.
    """
    input_pattern = re.compile(r'\\(input|include)\{([^\}]+)\}')

    def replacer(match):
        command, file_name = match.groups()
        file_path = os.path.join(base_path, f"{file_name}.tex")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return expand_inputs(base_path, file.read())
        else:
            # Return empty string if the file does not exist
            return ""
    
    return re.sub(input_pattern, replacer, file_content)

class ArXiv:
    def __init__(self, paperlink=None):
        if paperlink is not None:
            arxiv_id = paperlink.split('/')[-1]
            if arxiv_id.endswith("pdf"):
                # Getting rid of pdf suffix.
                arxiv_id = arxiv_id[:-4]

            self2 = ArXiv.search_arxiv([arxiv_id])[0]
            for k, v in self2.__dict__.items():
                setattr(self, k, v)

    def download_latex(self):
        arxiv_id = self.arxiv_id
        output_path = f"./{arxiv_id}"

        if not os.path.exists(output_path):
            source_link = "https://arxiv.org/e-print/" + arxiv_id
            response = requests.get(source_link)
            filename = arxiv_id + ".tar.gz"
            with open(filename, "wb") as f:
                f.write(response.content)

            untar(filename, output_path)
            # delete all non .tex files in all the root folders and subfolders to save space
            # and put all tex files into a long text
            # Traverse recursively  
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if not file.endswith(".tex") and not file.endswith(".bbl"):
                        os.remove(os.path.join(root, file))
                        
            # also remove the tar.gz file
            os.remove(filename)

        # Then we check which file is the main one. For this we check a .bbl file and start with the corresponding .tex file. 
        # Find the .bbl file
        main_tex = None
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(".bbl"):
                    main_tex = file[:-4] + ".tex"
                    break

        if main_tex is None:
            # If no .bbl file is found, we just take the first .tex file we find
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.endswith(".tex"):
                        main_tex = file
                        break

        # Then we do recursive expansion. 
        with open(os.path.join(output_path, main_tex), 'r') as f:
            content = f.read()
            all_content = expand_inputs(output_path, content)

        # Placeholder for introduction retrieval
        all_content += r"\section{dummy}"

        # title = arxiv_info["title"] 
        # abstract = arxiv_info["abstract"]

        # Then extract each section / subsection to get an idea on what's going on in details. 

        # Save all content
        # with open(f"{arxiv_id}.tex", "w") as f:
        #     f.write(all_content)

        sections = dict()
        introduction = ""

        for m in re.finditer(r'\\section{(.*?)}(.*?)\\section{', all_content, re.DOTALL):
            sec_title = m.group(1)
            sec_content = m.group(2)

            if sec_title == "Introduction":
                introduction = sec_content
            sections[sec_title] = sec_content

        self.all_content = all_content
        self.introduction = introduction
        self.sections = sections 

    @staticmethod
    def search_arxiv(keywords):
        url = f"https://export.arxiv.org/api/query?search_query=all:{'+'.join(keywords)}&start=0&max_results=10&sortBy=relevance&sortOrder=descending"
        response = requests.get(url)

        # use BeautifulSoup to parse the response
        parser = BeautifulSoup(response.text, features="xml")

        # extract title, summary and authors of the arXiv paper
        entries = parser.find_all("entry")
        all_papers = [] 
        for entry in entries:
            paperlink = entry.id.text
            title = entry.title.text
            summary = entry.summary.text
            # For all authors, extract the name
            authors = entry.find_all("author")
            authors = [author.find("name").text for author in authors]
            arxiv_id = paperlink.split("/")[-1]

            title = title.replace("\n"," ").replace("  ", " ") 
            summary = summary.replace("\n"," ").replace("  ", " ")

            paper = ArXiv()
            paper.arxiv_id = arxiv_id
            paper.title = title
            paper.abstract = summary
            paper.authors = authors
            paper.link = paperlink

            all_papers.append(paper)

        return all_papers


def paper2message(all_papers):
    for paper in all_papers:
        # send the title, summary and authors of the paper as text form
        title = paper['title']
        abstract = paper['abstract']
        link = paper["link"]
        arxiv_id = paper["arxiv_id"]

        yield f"<b>Title:</b> <a href='{link}'>{title}</a> ({arxiv_id}) \n<b>Authors:</b> {', '.join(paper['authors'])}\n\n<b>Abstract:</b> {abstract}\n"
        if "summary" in paper:
            if isinstance(paper["summary"], str):
               yield f"<b>Summary:</b> {paper['summary']}" 
            else:
                for msg in paper["summary"]:
                    yield msg