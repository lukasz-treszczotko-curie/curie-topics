This is a refined, production-ready execution plan designed for your scale and technical stack. It shifts the core logic from geometric embedding clustering to **Graph-based Community Detection**, which better captures the structural relationships of scientific concepts.

### **Phase 1: Data Ingestion & Infrastructure**
**Goal:** Efficiently load and filter the massive OpenAlex dataset without API bottlenecks.

*   **Task 1.1: Environment Setup**
    *   **Tools:** Python, `polars` (for fast DataFrame ops), `boto3`.
    *   **Compute:** High-RAM instance (e.g., 64GB+) or a Spark cluster if available.
*   **Task 1.2: Snapshot Acquisition**
    *   Connect to the OpenAlex S3 bucket (free access).
    *   Download the `works` and `topics` entities in **Parquet** format.
    *   *Optimization:* Do not download the entire dump if possible; use partitioned reads to select columns: `id`, `title`, `abstract_inverted_index`, `keywords`, `primary_topic`.
*   **Task 1.3: Partitioning**
    *   Filter works to those published in the last 10 years (to ensure subtopics are current).
    *   Group works by `primary_topic.id`. Save these as sharded Parquet files (e.g., `topic_id=T123/works.parquet`) to allow parallel processing later.

### **Phase 2: Vocabulary Analysis & Expansion**
**Goal:** Ensure there is enough keyword density to form meaningful clusters.

*   **Task 2.1: Keyword Density Audit**
    *   For each Topic (~4,500), calculate:
        *   `N_works`: Total works count.
        *   `N_keywords`: Unique OpenAlex keywords assigned.
        *   `Avg_keywords_per_work`: Density metric.
*   **Task 2.2: Conditional Keyword Extraction (The "Sparse Topic" Fix)**
    *   **Logic:** If a topic has `< 50` unique keywords or poor coverage, trigger an extraction pipeline.
    *   **Method:**
        1.  Reconstruct text from `abstract_inverted_index`.
        2.  Run **KeyBERT** or a fast **TF-IDF** pass over the topic's corpus.
        3.  Extract top 20 phrases per paper and append them to the existing OpenAlex keywords list for that work.
    *   *Why:* "26,000 keywords" globally is small. Many niche topics will be empty without this step.
*   **Task 2.3: Synonym Normalization**
    *   Use a lightweight embedding model (`all-MiniLM-L6-v2`) to merge near-duplicates within a topic (e.g., "ConvNet" $\to$ "Convolutional Neural Network") to density the graph.

### **Phase 3: Graph Construction & Community Detection**
**Goal:** Group keywords based on how they are actually used together in papers (co-occurrence).

*   **Task 3.1: Construct Topic-Level Graphs**
    *   **Tools:** `networkx` or `igraph` (C++ backend, much faster).
    *   **Nodes:** Keywords.
    *   **Edges:** Created when two keywords appear in the same work.
    *   **Edge Weight:** `Count(Works_with_A_and_B)`. Optional: Normalize using Jaccard Similarity to prevent common words from dominating.
*   **Task 3.2: Community Detection (The Clustering Step)**
    *   **Algorithm:** Apply the **Leiden Algorithm** (superior to Louvain for disconnected communities).
    *   **Configuration:** Optimize modularity.
    *   **Output:** A mapping of `Keyword_ID` $\to$ `Cluster_ID`.
*   **Task 3.3: Filter Noise**
    *   Discard "Singleton" clusters (clusters with only 1-2 keywords) or merge them into the nearest large cluster based on edge weight.

### **Phase 4: Agentic Subtopic Labeling**
**Goal:** Generate interpretable, hierarchical names for the clusters using LLMs.

*   **Task 4.1: Context Retrieval**
    *   For each identified Cluster (Subtopic Candidates), retrieve:
        1.  **Top 10 Keywords:** By PageRank/Degree Centrality within the cluster subgraph.
        2.  **Top 3 Representative Works:** Find works that contain the highest percentage of this cluster's keywords. Retrieve their **Titles**.
*   **Task 4.2: LLM Prompting Strategy**
    *   **Model:** GPT-4o-mini or Claude 3.5 Haiku (cost-effective for 15,000+ calls).
    *   **Prompt Structure:**
        > **Context:** You are categorizing research in the field of "{Topic Name}".
        > **Input:**
        > *   Keywords: [Key1, Key2, Key3...]
        > *   Representative Papers: ["Title A...", "Title B..."]
        > **Task:** Create a subtopic name (max 4 words).
        > **Constraint:** The name must be specific to the keywords but fit under "{Topic Name}".
*   **Task 4.3: Output Parsing**
    *   Store result as `Subtopic_Label`.

### **Phase 5: Validation & Integration**
**Goal:** Ensure the subtopics are accurate and map works correctly.

*   **Task 5.1: Automated "Judge" Verification**
    *   Sample 1 random work per generated subtopic.
    *   **Prompt:** "Does the paper '{Title}' fit the category '{Subtopic Label}'? Answer Yes/No."
    *   **Threshold:** Discard or flag subtopics with "No" responses.
*   **Task 5.2: Work Assignment**
    *   Assign every work in the Topic to a Subtopic.
    *   **Method:** A work belongs to Subtopic X if $>50\%$ of its keywords belong to Subtopic X's keyword cluster.
    *   *Note:* A work can belong to multiple subtopics (Soft Clustering).
*   **Task 5.3: Artifact Generation**
    *   Produce a final JSON/Parquet hierarchy:
        `Domain` $\to$ `Field` $\to$ `Subfield` $\to$ `Topic` $\to$ **`Subtopic`** $\to$ `Keywords`.

### **Tech Stack Recommendation**
| Component | Tool Recommendation |
| :--- | :--- |
| **Data Processing** | **Polars** (Python) - Essential for handling the Parquet files efficiently. |
| **Graph/Clustering** | **igraph** or **CDlib** - Optimized for community detection. |
| **Embeddings (for merging)** | **Sentence-Transformers** (`all-MiniLM-L6-v2`) - Fast and sufficient. |
| **LLM Inference** | **OpenAI API (Batch Mode)** or **vLLM** (if self-hosting Llama 3). |
| **Orchestration** | **Prefect** or simple Python scripts with `tqdm`. |