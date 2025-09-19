import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import config


def get_retriever(_docs, updated: bool):
    embed = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.isdir(config.PERSIST_DIR):
        db = Chroma.from_documents(
            _docs, embed, persist_directory=config.PERSIST_DIR)
    else:
        db = Chroma(persist_directory=config.PERSIST_DIR,
                    embedding_function=embed)
        if updated and _docs:
            db.add_documents(_docs)
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 50})
    return retriever, db


def build_chain(retriever, intents: list[str], k=3):
    llm = ChatOpenAI(model="gpt-4o", streaming=False)
    intents_desc = "\n".join(f"- {it}" for it in intents)
    modules_desc = "\n".join(
        f"- **{m}**: {d}" for m, d in config.MODULES_INFO.items())
    sys_prompt = (
        "You are a network-service composition assistant.\n\n"
        f"## Available Intents ({len(intents)})\n{intents_desc}\n\n"
        "## Module Library\n" + modules_desc + "\n\n"
        "## Output Format Rules\n" + config.OUTPUT_SPEC.format(k=k) + "\n\n"
        "## How to use feedback documents\n"
        "Pipelines labelled **Perfect**, keep exactly that pipeline for the same intent.\n"
        "**Partial**, try to remove unnecessary modules but keep the core order.\n"
        "**Bad**, never propose that module combination again for this intent.\n"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt + "\n{context}"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


def run_intent(intent: str, rag_chain) -> str:
    ans = rag_chain.invoke({"input": intent})["answer"]
    return ans
