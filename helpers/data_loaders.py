import os
import re
import json
import hashlib
import wikipediaapi
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader, PyPDFLoader
from langchain.schema import Document as LCDoc

import config

wiki_client = wikipediaapi.Wikipedia(
    language='en',
    user_agent='AgenticAIDT/1.0 (your-name@example.com)'
)


def clean_text(txt: str) -> str:
    txt = re.sub(r'/gid\d+', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def save_hashes(dct):
    with open(config.HASH_FILE_PATH, 'w') as f:
        json.dump(dct, f)


def load_hashes() -> dict:
    if os.path.exists(config.HASH_FILE_PATH):
        with open(config.HASH_FILE_PATH, 'r') as f:
            return json.load(f)
    return {}


def fetch_wiki_text(title: str) -> str | None:
    page = wiki_client.page(title)
    if page.exists():
        return page.text
    print(f"⚠ Wikipedia page not found: {title}")
    return None


def load_docx(path: str):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents() -> tuple[list, bool]:
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    prev = load_hashes()
    curr, docs, updated = {}, [], False
    api_key = os.getenv("FIRECRAWL_API_KEY")

    if os.path.exists(config.URL_PATH):
        curr['website.txt'] = md5(config.URL_PATH)
        if curr['website.txt'] != prev.get('website.txt'):
            updated = True
            with open(config.URL_PATH) as f:
                urls = [u.strip() for u in f]
            for u in urls:
                for d in splitter.split_documents(FireCrawlLoader(api_key, u, "scrape").load()):
                    d.metadata = {"source": u}
                    docs.append(d)

    if os.path.exists(config.WIKIPEDIA_PATH):
        with open(config.WIKIPEDIA_PATH) as f:
            for link in (l.strip() for l in f if l.strip()):
                title = link.split('/wiki/')[-1]
                if title in prev:
                    curr[title] = prev[title]
                    continue

                updated = True
                text = fetch_wiki_text(title)
                if text:
                    curr[title] = hashlib.md5(text.encode()).hexdigest()
                    for d in splitter.split_text(text):
                        docs.append(
                            LCDoc(page_content=d, metadata={"source": link}))

    def load_bulk(folder: str):
        nonlocal updated
        if not os.path.isdir(folder):
            return
        for fn in os.listdir(folder):
            if not fn.endswith((".pdf", ".docx")):
                continue
            path = os.path.join(folder, fn)
            curr[fn] = md5(path)
            if curr[fn] == prev.get(fn):
                continue
            updated = True
            try:
                if fn.endswith(".pdf"):
                    pieces = PyPDFLoader(path).load()
                else:
                    pieces = [LCDoc(page_content=load_docx(
                        path), metadata={"source": fn})]
                for d in splitter.split_documents(pieces):
                    d.page_content = clean_text(d.page_content)
                    d.metadata = {"source": fn}
                    docs.append(d)
            except Exception as e:
                print(f"⚠ Error processing file {path}: {e}")
                continue

    load_bulk(config.ORAN_DIR)
    load_bulk(config.GPP_DIR)
    load_bulk(config.PAPER_DIR)
    load_bulk(config.DELIVER_DIR)

    if updated:
        save_hashes(curr | prev)
    return docs, updated
