<div align="center">
  <h1>🧠 Advanced RAG Multi-Document Chatbot</h1>
  <p><i>An intelligent, self-correcting chatbot that reads your documents and answers questions accurately without making things up.</i></p>
</div>

---

## 📖 Overview

Imagine having a smart assistant that has read every PDF, Word document, and spreadsheet in your company. When you ask it a question, it doesn't just guess—it finds the exact paragraphs, reads them, and gives you a precise answer with citations. 

This project is an **Advanced Retrieval-Augmented Generation (RAG) System**. Standard AI models often hallucinate (make up facts) because they rely on their general training. This system forces the AI to only answer using **your specific documents**, making it highly reliable for business, research, and personal use.

---

## 🎯 The Problem it Solves (And Drawbacks Overcome)

Standard AI chatbots have several known flaws. We built specific components into this project to overcome each one:

1. **The "Missing Keyword" Problem**
   * *Drawback:* Standard AI search (Vector Search) is great at understanding *meaning*, but terrible at finding *exact names or IDs*.
   * *Solution:* We use **Hybrid Search**. We combine standard meaning-based search (Qdrant) with exact-keyword search (BM25). You get the best of both worlds.

2. **The "Irrelevant Information" Problem**
   * *Drawback:* Sometimes the search engine grabs documents that vaguely match but don't actually contain the answer. This confuses the AI.
   * *Solution:* We use a **Cross-Encoder Reranker**. It acts like a strict editor, double-checking the retrieved documents and throwing out the irrelevant ones *before* the AI sees them.

3. **The "Hallucination" Problem**
   * *Drawback:* If the AI can't find the answer in the documents, it will often lie and make one up to sound helpful.
   * *Solution:* We use a **Corrective LangGraph Workflow**. The system literally "grades" the documents. If they don't contain the answer, it rewrites your search and tries again. If it still fails, it safely admits: *"I don't have enough information to answer."*

4. **The "Amnesia" Problem**
   * *Drawback:* If you ask "Who is the CEO?", the bot says "Jane Doe". If you then ask "Where did she go to college?", the bot forgets who "she" is.
   * *Solution:* We use a **Conversational Query Rewriter**. It looks at your chat history and silently rewrites your follow-up into *"Where did Jane Doe go to college?"* before searching.

5. **The "Slow Re-upload" Problem**
   * *Drawback:* In basic systems, adding one new document means you have to re-process the entire library, wasting time and computing power.
   * *Solution:* We use a **Smart Indexer**. It creates a unique fingerprint for every paragraph. It only processes brand-new or modified files.

---

## 🏗️ How It Works (The Components)

Even if you aren't a programmer, here is the journey of your documents and your questions:

### 1. Document Ingestion & Chunking
* **What it does:** Reads PDFs, Word files, Excels, CSVs, and Text files. It chops them into small, overlapping "chunks" (paragraphs). 
* **Why:** The AI can't read a 500-page book in one go. Chunks make the text bite-sized and digestible.

### 2. Embeddings & Vector Store (BGE-M3 & Qdrant/You can choose yours)
* **What it does:** Translates human text into numbers (vectors) and stores them in a database.
* **Why:** This allows the system to search by *concept and meaning*, not just exact words.

### 3. The Brain (Qwen3-4B-Instruct/You can choose yours)
* **What it does:** A highly efficient, open-source AI model that reads the retrieved paragraphs and writes the final answer for you.
* **Why:** We run this in "4-bit quantization", which is a clever math trick that allows a powerful AI to run on a standard, affordable graphics card (GPU) without losing its smarts.

### 4. The Agent Workflow (LangGraph)
* **What it does:** The traffic controller. It routes your question, triggers the search, grades the results, and commands the AI to generate the answer.
* **Why:** It guarantees quality control and self-correction.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A GPU (NVIDIA T4 or better is recommended for the AI model)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ak0000007/Advance_RAG_MultiDoc_Chatbot.git
   cd Advance_RAG_MultiDoc_Chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install langgraph
   ```

3. **Add your documents:**
   Place your PDFs, Word documents, CSVs, or text files into the `data/` folder.

### Running on Google Colab (Recommended for Free GPU)
We have provided a complete test suite that sets everything up automatically.
1. Upload the `notebooks/05_full_pipeline_test.py` script to Google Colab.
2. Change your runtime to **T4 GPU**.
3. Run the script! It will automatically download the AI, index the test documents, and run quality checks.

---

## 📂 Project Structure

- `data/` - Put your raw documents here.
- `src/` - The core engine.
  - `ingestion/` - Code to read files.
  - `chunking/` - Code to chop text into paragraphs.
  - `embeddings/` & `vector_stores/` - Code to translate and store meaning.
  - `retrieval/` & `reranking/` - Code to search and filter the best results.
  - `llm/` - Code to run the AI model.
  - `graph/` - The logic that handles self-correction and grading.
- `notebooks/` - Contains the `05_full_pipeline_test.py` for testing everything on Colab.

---

## 🛠️ Built With

* [LangChain](https://python.langchain.com/) - Framework for AI applications
* [LangGraph](https://python.langchain.com/docs/langgraph) - Framework for agentic workflows and self-correction
* [Qdrant](https://qdrant.tech/) - High-performance Vector Database
* [HuggingFace](https://huggingface.co/) - Providing the BGE Embeddings, Reranker, and Qwen3 LLM
* [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) - Standard keyword search algorithm

---

## 📜 License

This project is open-source and currently  under Development. Feel free to fork, modify, and use it for your own data!
