# 🔧 INSTALL FIRST (only once):
# pip install biopython tqdm

from Bio import Entrez
from tqdm import tqdm
from time import sleep
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import pickle
import re

# === Configuration ===
ENTREZ_EMAIL = "rohannambiar370@gmail.com"
ENTREZ_API_KEY = "d7d72d403bc1a992b88d7c8befe734858708"  # Replace with your own Entrez API key
Entrez.email = ENTREZ_EMAIL
Entrez.api_key = ENTREZ_API_KEY

CONDITION = "frontal lobe edema"

def slugify(text):
    return re.sub(r'\W+', '_', text.lower()).strip('_')

slug = slugify(CONDITION)
INDEX_PATH = f"faiss_index_{slug}"
CACHE_PATH = f"cached_articles_{slug}.pkl"

# === 1. Dynamic Query Generation via LLM ===
query_prompt = PromptTemplate.from_template("""
You are an assistant that generates intelligent PubMed search queries from medical cases.

Condition: {condition}
Generate 3 diverse PubMed search queries relevant to its treatment, pathology, and imaging. Use concise phrasing.
Return each query as a new line.
""")

llm_for_prompt = Ollama(model="mistral")
generate_queries_chain = LLMChain(prompt=query_prompt, llm=llm_for_prompt)

print("\n📡 Generating PubMed search queries...")
query_output = generate_queries_chain.run(CONDITION)
search_queries = [q.strip() for q in query_output.strip().split('\n') if q.strip()]
print("\n🔎 Queries generated:")
for q in search_queries:
    print("  •", q)

# === 2. Fetch Articles ===
def fetch_pubmed_articles(queries, max_results_per_query=30):
    articles = []
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results_per_query)
            record = Entrez.read(handle)
            pmids = record["IdList"]
            for pmid in tqdm(pmids, desc=f"Fetching {len(pmids)} articles"):
                try:
                    fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
                    data = Entrez.read(fetch_handle)
                    article = data["PubmedArticle"][0]["MedlineCitation"]["Article"]
                    title = article.get("ArticleTitle", "")
                    abstract = article.get("Abstract", {}).get("AbstractText", [""])[0]
                    text = f"{title}\n\n{abstract}".strip()
                    if text:
                        articles.append(text)
                except Exception:
                    continue
                sleep(0.3)  # With API key: 10 req/sec
        except Exception:
            continue
    return list(set(articles))  # Deduplicate

# === 3. Load or Fetch and Cache ===
print("\n🔍 Checking for cached articles...")
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        texts = pickle.load(f)
    print(f"✅ Loaded {len(texts)} cached articles.")
else:
    print("🔍 Fetching abstracts from PubMed...")
    texts = fetch_pubmed_articles(search_queries)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(texts, f)
    print(f"✅ Fetched and cached {len(texts)} articles.")

# === 4. Chunk & Embed ===
print("⚙️  Chunking & embedding...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = [Document(page_content=chunk) for text in texts for chunk in splitter.split_text(text)]

embedding_model = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

if os.path.exists(INDEX_PATH):
    print("💾 Loading existing FAISS index...")
    vector_store = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("⚖️ Building FAISS index...")
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(INDEX_PATH)
    print("✅ FAISS index saved locally.")

# === 5. RAG Setup ===
print("🤖 Launching RAG pipeline...")
llm = Ollama(model="mistral")  # Uses GPU if Ollama is running on GPU backend
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What are the imaging features, treatment considerations, and mass effects associated with frontal lobe edema?"
docs = retriever.get_relevant_documents(query)

print("\n===== Retrieved Documents =====")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Doc {i} ---\n{doc.page_content[:500]}...")

response = qa.invoke(query)
print("\n===== LLM Response =====")
print(response)
