import os
import json
import time
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ********************* open api key **********************

os.environ["OPENAI_API_KEY"] = "*************************"

# ************** embedding model and large langauge model name *****************************
embedding_model_name = "text-embedding-ada-002"
llm_model_name = "gpt-3.5-turbo-16k"

# initialize the embeddings using openAI ada text embedding library and the llm model using gpt-3.5-turbo-16k
embeddings = OpenAIEmbeddings(model=embedding_model_name)
llm = OpenAI(temperature=0, model_name=llm_model_name)


def read_pdfs_log(directory_path, file_name="pdfs_log.json"):
    file_path = os.path.join(directory_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            return json.load(fp)["saved_pdfs"]
    return None

def get_all_pdfs_names(directory_path):
    return [
        filename
        for filename in os.listdir(directory_path)
        if filename.endswith(".pdf")
    ]

def save_pdfs_log(processed_pdfs, directory_path, file_name="pdfs_log.json"):
    pdfs_log = {"saved_pdfs": processed_pdfs}
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, "w") as fp:
        json.dump(pdfs_log, fp)

def preprocess_texts(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def read_titles_and_dois(file_path):
    with open(file_path, "r") as fp:
        return json.load(fp)

def read_pdf_text(path, title, doi, preprocess_langchain=False):
    reader = PdfReader(path)
    raw_text = f"Title: {title}\nDOI: {doi}\n\n"
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    if preprocess_langchain:
        texts = preprocess_texts(raw_text)
    else:
        texts = raw_text
    return texts

def process_all_pdfs(directory_path, pdfs_to_process, titles_and_dois, preprocess_langchain=False):
    all_texts = []
    for filename in pdfs_to_process:
        filepath = os.path.join(directory_path, filename)
        title = titles_and_dois.get(filename, {}).get("title", "Unknown Title")
        doi = titles_and_dois.get(filename, {}).get("doi", "Unknown DOI")
        texts = read_pdf_text(filepath, title, doi, preprocess_langchain)
        all_texts.extend(texts)
    return all_texts

def embed_texts(texts, delay=60, max_length=1000000):
    embeddings = OpenAIEmbeddings()
    samples_texts = []
    sample_texts_len = 0
    docstores_count = 0
    docstore = None
    for text in texts:
        text_len = len(text)
        if sample_texts_len + text_len <= max_length:
            samples_texts.append(text)
        else:
            sample_docstore = FAISS.from_texts(samples_texts, embeddings)
            if docstores_count == 0:
                docstore = sample_docstore
            else:
                docstore.merge_from(sample_docstore)

            samples_texts = [text]
            sample_texts_len = 0
            time.sleep(delay)
            print(f"created {docstores_count + 1} docstore")
            docstores_count += 1

        sample_texts_len += text_len

    # Remaining text
    sample_docstore = FAISS.from_texts(samples_texts, embeddings)
    if docstores_count == 0:
        docstore = sample_docstore
    else:
        docstore.merge_from(sample_docstore)
    print(f"created {docstores_count + 1} docstore")

    return docstore

def create_docstore(faiss_docstore_path, docstore_path, titles_and_dois_path):
    preprocessed_pdfs = read_pdfs_log(docstore_path, file_name="pdfs_log.json")
    preprocessed_pdfs = set(preprocessed_pdfs) if preprocessed_pdfs else set()
    all_pdfs = get_all_pdfs_names(docstore_path)
    pdfs_to_process = set(all_pdfs) - preprocessed_pdfs
    titles_and_dois = read_titles_and_dois(titles_and_dois_path)
    texts = process_all_pdfs(
        docstore_path, pdfs_to_process, titles_and_dois, preprocess_langchain=True
    )
    embeddings = OpenAIEmbeddings()
    docstore = embed_texts(texts, delay=60, max_length=2000000)
    if os.path.exists(faiss_docstore_path):
        docstore_old = FAISS.load_local(faiss_docstore_path, embeddings)
        docstore.merge_from(docstore_old)
    docstore.save_local(faiss_docstore_path)
    save_pdfs_log(all_pdfs, docstore_path, file_name="pdfs_log.json")

if __name__ == "__main__":
    faiss_docstore_path = ""     # Path where you want to create faiss doc store
    docstore_path = ""          # Path for your document store
    titles_and_dois_path = ""  # Path to the JSON file containing titles and DOIs
    create_docstore(faiss_docstore_path, docstore_path, titles_and_dois_path)
    print(f"Doc store saved")

















