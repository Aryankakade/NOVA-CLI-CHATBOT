from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # type: ignore
from git import Repo # type: ignore
from dotenv import load_dotenv # type: ignore
import os
import shutil
import stat

# --- Load environment variables ---
load_dotenv()

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TEMP_REPO_PATH = "./temp_repo"

SUPPORTED_EXTENSIONS = [
    '.py', '.md', '.ipynb', '.js', '.html', '.css', '.txt', '.sh', '.yml', '.yaml',
    '.json', '.xml', '.config', '.ini'
]

def remove_readonly(func, path, excinfo):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def filter_complex_metadata(metadata):
    # Only allow Chroma-compatible metadata values
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_meta[k] = v
        elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            clean_meta[k] = v[0]  # convert ['eng'] to 'eng'
        else:
            clean_meta[k] = str(v)  # fallback: convert to string
    return clean_meta

def ingest_github_repo(repo_url):
    print(f"Cloning repository: {repo_url} into {TEMP_REPO_PATH}")
    if os.path.exists(TEMP_REPO_PATH):
        shutil.rmtree(TEMP_REPO_PATH, onerror=remove_readonly)
    Repo.clone_from(repo_url, to_path=TEMP_REPO_PATH)
    print("Loading documents from repository...")
    documents = []
    for root, _, files in os.walk(TEMP_REPO_PATH):
        if '.git' in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in SUPPORTED_EXTENSIONS:
                try:
                    print(f"  - Processing: {file}")
                    loader = UnstructuredFileLoader(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata['source'] = file_path.replace(TEMP_REPO_PATH, repo_url)
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"  - Error processing file {file_path}: {e}")
            else:
                print(f"  - Ignoring (unsupported extension): {file}")
    shutil.rmtree(TEMP_REPO_PATH, onerror=remove_readonly)
    return documents

def process_and_store_documents(documents):
    if not documents:
        print("No documents were loaded to process.")
        return
    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_documents)} text chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print(f"Creating/updating Chroma vector store at: {CHROMA_DB_PATH}...")

    # Filter complex metadata for each chunk
    for doc in chunked_documents:
        doc.metadata = filter_complex_metadata(doc.metadata)

    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    vector_store.persist()
    print(f"✅ Ingested {len(chunked_documents)} chunks into Chroma.")
    print("\n--- Document Ingestion Complete! ---")

def main(source_path: str, enhanced_processing: bool = False, include_file_contents: bool = False):
    if source_path.startswith("http") and "github.com" in source_path:
        print("Using Chroma for vector storage.")
        if os.path.exists(CHROMA_DB_PATH):
            print("Deleting old Chroma knowledge base...")
            shutil.rmtree(CHROMA_DB_PATH)

        # ✅ Repo ingestion
        docs = ingest_github_repo(source_path)
        process_and_store_documents(docs)

        # ✅ Extra AI analysis if enabled
        if enhanced_processing:
            from backend import nova_system
            provider = nova_system.api_manager.get_best_provider()
            ai_response = None
            if provider:
                try:
                    import requests
                    resp = requests.post(
                        provider["url"],
                        headers=provider["headers"](),
                        json={
                            "model": provider["models"][0],
                            "messages": [{
                                "role": "user",
                                "content": f"Analyze this GitHub repo: {source_path}. "
                                           f"Provide repository analysis, code quality review, and debugging suggestions."
                            }]
                        },
                        timeout=60
                    )
                    if resp.ok:
                        ai_response = resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    ai_response = f"⚠️ AI analysis failed: {e}"

            return {"docs_ingested": len(docs), "ai_analysis": ai_response}

        return {"docs_ingested": len(docs)}

    else:
        return {"error": "Currently only GitHub repository URLs are supported."}
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <github_repo_url>")
        sys.exit(1)
    repo_url = sys.argv[1]
    main(repo_url)