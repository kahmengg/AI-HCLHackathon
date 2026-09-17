# Wealth Advisor Assistant

An agentic Retrieval-Augmented Generation (RAG) prototype built for a wealth-management use case.

The assistant combines structured client and portfolio data with unstructured documents such as fund factsheets, policy documents, RM call notes, client correspondence, and complaint letters.

It uses function calling, semantic retrieval, CrossEncoder reranking, and grounded LLM generation to answer wealth-management questions using evidence from the supplied dataset.

---

## Features

### Structured Data Retrieval

Structured client information is stored in SQLite and accessed through Python tools.

Supported information includes:

- Client profiles
- Risk profiles and risk scores
- Portfolio holdings
- Asset allocation
- Transaction history
- Concentration checks
- Potential suitability mismatches

Structured data is used when exact values or records are required.

---

### Unstructured Document Retrieval

Unstructured documents are processed through a RAG pipeline:

1. Load PDFs and JSON correspondence
2. Extract and clean text
3. Split documents into chunks
4. Generate embeddings
5. Store embeddings in ChromaDB
6. Retrieve semantically relevant chunks
7. Rerank candidate chunks using a CrossEncoder
8. Provide the strongest evidence to the LLM

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2


The reranking model is:
cross-encoder/ms-marco-MiniLM-L-6-v2
Agentic Tool Calling
The assistant is implemented as a tool-calling agent.
Rather than using one retrieval method for every question, the LLM can choose from tools such as:
- Client profile retrieval
- Portfolio retrieval
- Transaction retrieval
- Suitability mismatch detection
- Document retrieval
For example, a suitability question may require both:
SQLite
→ exact client risk profile and portfolio holdings
and:
ChromaDB
→ RM notes, complaints, policies, and correspondence
The agent combines the returned evidence before generating its answer.
Grounded Responses
The LLM is instructed to operate only on information returned by the available tools.
The system is designed to:
- Avoid unsupported factual claims
- Separate retrieved facts from interpretation
- Avoid using external financial or regulatory knowledge as evidence
- Avoid inventing investment actions or next steps
- Indicate when available evidence is insufficient
- Reference supporting sources
This reduces hallucination risk and improves traceability.
Retrieval Reranking
Initial semantic retrieval is performed using ChromaDB.
Because vector similarity alone does not always rank the most useful evidence first, the system retrieves a candidate pool and applies a CrossEncoder reranker.
Current default retrieval flow:
User Question
      |
      v
Embedding
      |
      v
ChromaDB
      |
      v
Top 10 Candidate Chunks
      |
      v
CrossEncoder Reranker
      |
      v
Top 5 Evidence Chunks
      |
      v
LLM
Retrieval Evaluation
A labelled evaluation set is used to measure whether expected evidence is retrieved.
The primary metric is Recall@K.
- Recall@1 measures how much expected evidence appears in the first result.
- Recall@3 measures how much expected evidence appears within the top three results.
- Recall@5 measures how much expected evidence appears within the top five results.
Evaluation Results
Retrieval Method	Recall@1	Recall@3	Recall@5
Baseline Vector Retrieval	6.67%	35.00%	60.00%
CrossEncoder Reranking	18.33%	65.00%	80.00%
Query Rewrite + Reranking	28.33%	65.00%	70.00%
Query Fusion + Reranking	18.33%	65.00%	80.00%
Document-Type Routing + Reranking	18.33%	65.00%	80.00%


CrossEncoder reranking increased:
Recall@3: 35% → 65%
Recall@5: 60% → 80%
Reranking was therefore selected as the default retrieval strategy because it achieved the strongest overall recall while keeping the runtime pipeline relatively simple.
Query Rewriting Experiments
The project also implements LLM-based query rewriting.
Example:
Original:
Does Park Ji-hoon's 35% portfolio allocation to the APEX
Autocallable Note breach firm policy?

Rewritten:
CL011 Park Ji-hoon 35% APEX Autocallable Note policy breach
Query rewriting improved Recall@1 in testing but did not improve overall Recall@5.
It is therefore retained as an experimental feature but is not enabled by default.
Query Fusion
Query fusion searches using both:
Original Query
+
LLM-Rewritten Query
The two candidate sets are merged, deduplicated, and reranked.
Although this produced competitive results, it did not outperform the simpler reranking-only pipeline on the evaluation set.
It is therefore retained as an experimental capability.
Document-Type Routing
Document-type-aware retrieval was also tested.
For example, a question containing terms such as:
policy
breach
suitability
risk profile
compliance
can trigger an additional search restricted to policy documents.
The resulting candidates are merged with global retrieval candidates before reranking.
This capability was implemented successfully but did not produce a measurable Recall@K improvement over standard reranking on the current evaluation set.
Technology Stack
Backend
- Python
- SQLite
- ChromaDB
- Sentence Transformers
- CrossEncoder reranking
- Groq API
Models
Embedding:
sentence-transformers/all-MiniLM-L6-v2
Reranking:
cross-encoder/ms-marco-MiniLM-L-6-v2
LLM:
openai/gpt-oss-20b
via Groq.
Frontend
- Streamlit
Project Structure
HCLHackathon/
│
├── data/
│   ├── wealth_advisor.db
│   └── chroma/
│
├── src/
│   │
│   ├── agent/
│   │   ├── advisor_agent.py
│   │   └── answer.py
│   │
│   ├── embedding/
│   │   └── embedder.py
│   │
│   ├── evaluation/
│   │   ├── eval_set.json
│   │   └── retrieval_eval.py
│   │
│   ├── frontend/
│   │   └── app.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── json_loader.py
│   │   └── pdf_loader.py
│   │
│   ├── llm/
│   │   └── groq_client.py
│   │
│   ├── receiver/
│   │   ├── query_rewriter.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   │
│   ├── structured/
│   │   ├── database.py
│   │   ├── ingest_structured.py
│   │   └── structured_data.py
│   │
│   ├── tools/
│   │   ├── rag_tools.py
│   │   └── structured_tools.py
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py
│   │
│   └── pipeline.py
│
├── .env
├── .gitignore
└── README.md
Setup
1. Create a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate
2. Install Dependencies
Install the required Python packages for the project.
Example:
pip install -r requirements.txt
3. Configure Environment Variables
Create:
.env
and add:
GROQ_API_KEY=your_api_key_here
Do not commit .env to version control.
Build Structured Data
Run the structured-data ingestion process:
python -m src.structured.ingest_structured
This creates the SQLite wealth-management database.
Build the Vector Database
Run:
python -m src.pipeline
The pipeline:
Documents
   ↓
Extraction
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
ChromaDB
The current dataset produces approximately 97 stored chunks.
Run the Retrieval Evaluation
python -m src.evaluation.retrieval_eval
This evaluates the retrieval strategies using the labelled evaluation set.
The selected production configuration is:
use_reranker=True
candidate_k=10

use_query_rewrite=False
use_query_fusion=False
use_doc_type_routing=False

Run Agent Test
python -m src.test.test_agent
Example question:
Does Robert Chua's APEX autocallable note raise any
suitability concerns?
The agent can combine structured client information with retrieved document evidence before generating its answer.
Run the Application
From the project root:
python -m streamlit run src/frontend/app.py
The Streamlit interface provides:
- Wealth-management question answering
- Agentic tool calling
- Structured data access
- Reranked semantic retrieval
- Grounded responses
- Retrieved evidence display
- Source metadata
- Retrieval scores
Example Queries
Suitability Analysis
Does Robert Chua's APEX autocallable note raise any
suitability concerns?
This may combine:
Client profile
+
Portfolio holdings
+
RM call notes
+
Client correspondence
+
Complaint evidence
Regulatory / Remittance Analysis
How much LRS remittance headroom does Arjun Mehta have left,
and can his planned USD 60,000 top-up proceed as-is?
Client Request Timeline
Walk me through what has happened with James Sullivan's
request to de-risk his portfolio ahead of retirement.
Has any rebalancing occurred yet?
Fund Factsheet Retrieval
What is the Summary Risk Indicator of the APAC Stable Income
Money Market Fund and what is the minimum initial investment?
Design Decisions
Why SQLite and ChromaDB?
The project separates structured and unstructured information.
SQLite is used for exact data such as:
Risk Score = 2
AUM = SGD 850,000
Allocation = 15%
ChromaDB is used for semantic information such as:
"The client did not fully understand how the note worked."
This avoids forcing exact structured records through semantic search.
Why Reranking?
Embedding similarity is fast but does not always rank the strongest evidence first.
A CrossEncoder can examine:
Question + Candidate Passage
together, allowing it to make a more precise relevance judgement.
The evaluation showed that this improved Recall@5 from:
60% → 80%
which justified using reranking in the default pipeline.
Why Not Enable Every Retrieval Technique?
Query rewriting, query fusion, and document-type routing were implemented and evaluated.
However, they did not consistently improve the main retrieval metric beyond the simpler reranking pipeline.
The final system therefore prioritises:
measured performance
+
simplicity
+
lower latency
rather than enabling additional components without evidence that they improve retrieval quality.
Limitations
The current system is a prototype built using synthetic wealth-management data.
Known limitations include:
- Small evaluation dataset
- Retrieval quality depends on document wording and embedding similarity
- CrossEncoder scores represent relative relevance and are not probabilities
- Some questions may require evidence distributed across multiple documents
- Source citation formatting can be further standardised
- The system is not intended to provide real financial, investment, legal, or regulatory advice
Disclaimer
All client records, portfolio data, documents, policies, transactions, correspondence, and scenarios used by this project are synthetic and intended solely for demonstration and evaluation purposes.
The application should not be relied upon for real investment, compliance, legal, or financial decisions.
