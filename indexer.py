"""
indexer.py
Handles: pulling a repo (local path or GitHub URL), walking its source files,
splitting them into language-aware chunks with accurate line-number metadata,
embedding those chunks, and storing them in a persistent Chroma vector store.
"""

import os
import shutil
import hashlib
import tempfile
import time
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# --- Extension -> LangChain Language mapping -------------------------------
EXT_LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".php": Language.PHP,
    ".rb": Language.RUBY,
    ".swift": Language.SWIFT,
    ".scala": Language.SCALA,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
}

# Extensions we index at all. Anything else (images, binaries, lockfiles) is skipped.
INDEXABLE_EXTS = set(EXT_LANGUAGE_MAP.keys()) | {".txt", ".json", ".yaml", ".yml", ".toml"}

# Never walk into these directories
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".pytest_cache", ".mypy_cache", "target",
    ".idea", ".vscode", "site-packages", "egg-info",
}

MAX_FILE_SIZE_BYTES = 400_000  # skip anything absurdly large (generated files, data dumps)


def resolve_source(source: str, workdir: str) -> str:
    """
    Accepts either a local folder path or a GitHub URL.
    Returns a local path to the code on disk.
    """
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://") or source.endswith(".git"):
        import git  # GitPython

        repo_hash = hashlib.sha1(source.encode()).hexdigest()[:10]
        dest = os.path.join(workdir, f"repo_{repo_hash}")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        git.Repo.clone_from(source, dest, depth=1)
        return dest

    if not os.path.isdir(source):
        raise ValueError(f"'{source}' is not a valid local directory or a GitHub URL.")
    return source


def _iter_source_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in INDEXABLE_EXTS:
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            yield full_path, ext


def _split_file_with_line_numbers(text: str, ext: str):
    """
    Splits file text into chunks and, for each chunk, computes the
    1-indexed [start_line, end_line] it corresponds to in the original file.
    This is what makes citations file-and-line grounded rather than just
    file-grounded.
    """
    language = EXT_LANGUAGE_MAP.get(ext)
    if language is not None:
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=1200, chunk_overlap=150
            )
        except Exception:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    chunks = splitter.split_text(text)

    # Map each chunk back to its line range by locating it in the source text.
    results = []
    search_from = 0
    for chunk in chunks:
        idx = text.find(chunk, search_from)
        if idx == -1:
            idx = text.find(chunk)  # fallback: search from the start
        if idx == -1:
            start_line, end_line = None, None
        else:
            start_line = text.count("\n", 0, idx) + 1
            end_line = start_line + chunk.count("\n")
            search_from = idx + 1  # allow overlap, but keep forward progress
        results.append((chunk, start_line, end_line))
    return results


def build_documents(repo_path: str, progress_callback=None):
    """Walks the repo and returns a list of langchain Document objects with rich metadata."""
    documents = []
    files = list(_iter_source_files(repo_path))
    for i, (full_path, ext) in enumerate(files):
        rel_path = os.path.relpath(full_path, repo_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if not text.strip():
            continue

        for chunk, start_line, end_line in _split_file_with_line_numbers(text, ext):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_path": rel_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": ext.lstrip("."),
                    },
                )
            )
        if progress_callback:
            progress_callback(i + 1, len(files), rel_path)
    return documents


def get_embeddings(use_gemini: bool, google_api_key: str = None):
    if use_gemini:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=google_api_key
        )
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(documents, persist_dir: str, embeddings):
    # IMPORTANT: persist_dir must be unique per repository/source (see get_persist_dir_for).
    # Chroma caches an internal "System" object in-process keyed by this exact path and
    # never evicts it. Deleting the folder and recreating a client at the *same* path
    # within the same running process leaves that cached System pointing at a database
    # that no longer exists on disk, causing "Could not connect to tenant default_tenant"
    # on the next index. Using a fresh path per repo sidesteps this entirely.
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    # Defensive: if this exact path was already used earlier in this process
    # (e.g. re-indexing the same repo twice), drop chromadb's cached System
    # for it so we don't reconnect to a stale in-memory reference.
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient._identifier_to_system.pop(persist_dir, None)
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name="codebase", embedding_function=embeddings, persist_directory=persist_dir
    )
    # Batched to stay under Google's 100-texts-per-call embedding limit (also fine
    # for local sentence-transformers embeddings, just means slightly more calls).
    batch_size = 100
    max_retries = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                # Free-tier embedding APIs (e.g. Gemini's 100 requests/minute cap)
                # commonly hit transient 429s on larger repos. Back off and retry
                # rather than failing the whole index over a rate limit.
                is_rate_limit = "429" in str(e) or "ResourceExhausted" in type(e).__name__
                if not is_rate_limit or attempt == max_retries - 1:
                    raise
                wait_s = min(60, 2 ** attempt * 5)
                time.sleep(wait_s)
    return vectorstore


def get_persist_dir_for(source: str) -> str:
    """A stable, unique Chroma persist directory per repo source (path or URL)."""
    source_hash = hashlib.sha1(source.encode()).hexdigest()[:10]
    return os.path.join(get_index_workdir(), f"chroma_{source_hash}")


def load_vectorstore(persist_dir: str, embeddings):
    return Chroma(
        collection_name="codebase", embedding_function=embeddings, persist_directory=persist_dir
    )


def get_index_workdir() -> str:
    d = os.path.join(tempfile.gettempdir(), "codebaseqa_index")
    os.makedirs(d, exist_ok=True)
    return d
