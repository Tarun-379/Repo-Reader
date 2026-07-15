"""
app.py
CodeCite: Ask Your Repository -- Streamlit front-end.

Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from indexer import (
    resolve_source, build_documents, get_embeddings,
    build_vectorstore, load_vectorstore, get_index_workdir, get_persist_dir_for,
)
from qa_chain import answer_question
from readme_generator import generate_readme_section

load_dotenv()

st.set_page_config(page_title="CodeCite", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------- session --
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧭 CodeCite")
    st.caption("Ask your repository where a feature lives.")

    google_api_key = st.text_input(
        "Google (Gemini) API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Free key: https://aistudio.google.com/app/apikey. "
        "Used for answering questions and the README generator.",
    )

    st.divider()
    source = st.text_input(
        "Local folder path OR GitHub URL",
        placeholder="/path/to/project  or  https://github.com/user/repo",
    )
    embedding_choice = st.radio(
        "Embeddings",
        ["Local (free, sentence-transformers)", "Gemini (needs API key)"],
        index=0,
        help="Local runs fully offline with no rate limits — recommended for indexing. "
        "Gemini's free tier caps embedding at 100 requests/minute, which most repos "
        "will exceed; the app will back off and retry automatically, but indexing "
        "will be slower.",
    )
    use_gemini_embeddings = embedding_choice.startswith("Gemini")

    if st.button("📥 Index repository", type="primary", use_container_width=True):
        if not source:
            st.error("Enter a local path or GitHub URL first.")
        elif use_gemini_embeddings and not google_api_key:
            st.error("Gemini embeddings need an API key.")
        else:
            with st.spinner("Resolving repository..."):
                try:
                    repo_path = resolve_source(source, get_index_workdir())
                except Exception as e:
                    st.error(f"Could not load repository: {e}")
                    repo_path = None

            if repo_path:
                progress_bar = st.progress(0, text="Reading files...")

                def _progress(done, total, current_file):
                    progress_bar.progress(
                        done / max(total, 1), text=f"Chunking {current_file} ({done}/{total})"
                    )

                with st.spinner("Splitting into chunks..."):
                    documents = build_documents(repo_path, progress_callback=_progress)

                if not documents:
                    st.error("No indexable source files found in that repository.")
                else:
                    with st.spinner(f"Embedding {len(documents)} chunks..."):
                        embeddings = get_embeddings(use_gemini_embeddings, google_api_key)
                        persist_dir = get_persist_dir_for(source)
                        vectorstore = build_vectorstore(documents, persist_dir, embeddings)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.repo_path = repo_path
                    st.session_state.chat_history = []
                    progress_bar.empty()
                    st.success(f"Indexed {len(documents)} chunks from {len(set(d.metadata['file_path'] for d in documents))} files.")

    if st.session_state.repo_path:
        st.info(f"Indexed: `{st.session_state.repo_path}`")

# -------------------------------------------------------------------- tabs
tab_chat, tab_readme = st.tabs(["💬 Ask the repo", "📄 Generate README"])

with tab_chat:
    if not st.session_state.vectorstore:
        st.write("Index a repository from the sidebar to get started.")
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if turn["sources"]:
                    with st.expander(f"📎 {len(turn['sources'])} source chunk(s)"):
                        for src in turn["sources"]:
                            loc = src["file_path"]
                            if src["start_line"]:
                                loc += f" (lines {src['start_line']}-{src['end_line']})"
                            st.markdown(f"**`{loc}`**")
                            st.code(src["snippet"], language=None)

        question = st.chat_input("e.g. Where is user authentication implemented?")
        if question:
            if not google_api_key:
                st.error("Add your Gemini API key in the sidebar to ask questions.")
            else:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Searching the codebase..."):
                        result = answer_question(
                            st.session_state.vectorstore, question, google_api_key
                        )
                    st.write(result["answer"])
                    if result["sources"]:
                        with st.expander(f"📎 {len(result['sources'])} source chunk(s)"):
                            for src in result["sources"]:
                                loc = src["file_path"]
                                if src["start_line"]:
                                    loc += f" (lines {src['start_line']}-{src['end_line']})"
                                st.markdown(f"**`{loc}`**")
                                st.code(src["snippet"], language=None)
                st.session_state.chat_history.append(
                    {"question": question, "answer": result["answer"], "sources": result["sources"]}
                )

with tab_readme:
    if not st.session_state.repo_path:
        st.write("Index a repository first.")
    else:
        st.write("Draft a README section directly from the indexed repository's structure.")
        if st.button("✍️ Generate README section"):
            if not google_api_key:
                st.error("Add your Gemini API key in the sidebar.")
            else:
                with st.spinner("Analyzing structure and drafting..."):
                    readme_text = generate_readme_section(
                        st.session_state.repo_path, google_api_key
                    )
                st.markdown(readme_text)
                st.download_button(
                    "⬇️ Download as README.md",
                    data=readme_text,
                    file_name="README_generated.md",
                    mime="text/markdown",
                )
