"""
qa_chain.py
Turns a user question into a grounded, file-and-line-cited answer using only
retrieved chunks from the indexed repository. No answer is allowed to rely
on the model's general knowledge of the codebase -- if it isn't in the
retrieved context, the model is instructed to say so.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a senior engineer answering questions about a specific codebase.

Rules you must follow:
1. Answer ONLY using the CONTEXT below. Do not use outside knowledge of similar
   libraries or frameworks, and do not guess at code that isn't shown.
2. Every claim about "where" something is implemented must reference the exact
   file path (and line numbers, if given) from the CONTEXT metadata.
3. If the answer is not present in the CONTEXT, say plainly that you could not
   find it in the indexed code, and suggest what to search for instead. Never
   fabricate a file path or function that is not shown in the CONTEXT.
4. When useful, quote the specific relevant lines (kept short) to justify the answer.

CONTEXT:
{context}
"""

USER_PROMPT = "Question: {question}"


def _format_context(source_documents):
    blocks = []
    for doc in source_documents:
        meta = doc.metadata
        loc = meta.get("file_path", "unknown")
        if meta.get("start_line"):
            loc += f" (lines {meta['start_line']}-{meta['end_line']})"
        blocks.append(f"### {loc}\n```{meta.get('language', '')}\n{doc.page_content}\n```")
    return "\n\n".join(blocks)


def get_llm(google_api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.1):
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=google_api_key, temperature=temperature
    )


def answer_question(vectorstore, question: str, google_api_key: str, k: int = 6):
    """Retrieves top-k relevant chunks and asks Gemini to answer strictly from them."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    source_documents = retriever.invoke(question)

    if not source_documents:
        return {
            "answer": "I couldn't find anything relevant to that in the indexed repository.",
            "sources": [],
        }

    llm = get_llm(google_api_key)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm
    context = _format_context(source_documents)
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": [
            {
                "file_path": d.metadata.get("file_path"),
                "start_line": d.metadata.get("start_line"),
                "end_line": d.metadata.get("end_line"),
                "snippet": d.page_content,
            }
            for d in source_documents
        ],
    }
