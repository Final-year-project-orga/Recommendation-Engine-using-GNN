# 🛒 E-commerce Recommendation Engine Using Graph Neural Networks

> **An end-to-end personalized e-commerce recommendation engine built using Graph Neural Networks (GNNs), graph-based collaborative filtering, candidate retrieval, ranking, and real-time recommendation serving.**

---

## 📌 Overview

This project implements a scalable **e-commerce recommendation engine** that uses **Graph Neural Networks (GNNs)** to model relationships between users, products, categories, brands, and user interactions.

Traditional recommendation systems often represent users and products as independent feature vectors. This project instead represents the e-commerce ecosystem as a **graph**, allowing the recommendation model to learn from the relationships and interaction patterns between entities.

The system learns personalized **user and product embeddings** and uses them to generate, filter, rank, and serve Top-N product recommendations.

### Core pipeline

```text
User Interaction Data
        │
        ▼
Data Preprocessing
        │
        ▼
Graph Construction
        │
        ▼
GNN / Graph Collaborative Filtering
        │
        ▼
User & Product Embeddings
        │
        ▼
Candidate Generation
        │
        ▼
Filtering
        │
        ▼
Ranking
        │
        ▼
Post-Processing
        │
        ▼
Personalized Top-N Recommendations
```

---

# 🎯 Objectives

The primary objectives of this project are:

* Build a complete recommendation engine for an e-commerce platform.
* Represent users and products as nodes in an interaction graph.
* Use GNN-based representation learning for personalized recommendations.
* Learn meaningful user and product embeddings.
* Implement efficient candidate retrieval.
* Rank candidates using learned and contextual features.
* Handle cold-start users and products.
* Evaluate the model using standard recommendation metrics.
* Compare GNN performance against traditional recommendation baselines.
* Provide a production-style API for recommendation serving.
* Design the system so the recommendation engine can operate independently from the e-commerce application.

---

# 🧠 Why Graph Neural Networks?

E-commerce interactions naturally form a graph.

For example:

```text
                 ┌─────────────┐
                 │    User     │
                 │    U101     │
                 └──────┬──────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
        VIEW          CART        PURCHASE
          │             │             │
          ▼             ▼             ▼
     ┌────────┐    ┌────────┐    ┌────────┐
     │Product │    │Product │    │Product │
     │  P101  │    │  P205  │    │  P501  │
     └────────┘    └────────┘    └────────┘
```

A user is connected to products through different types of interactions.

Products are also connected through metadata:

```text
Product
   │
   ├──── BELONGS_TO ────► Category
   │
   └──── HAS_BRAND ─────► Brand
```

A GNN can propagate information through these relationships and learn representations that capture both:

1. **User preferences**
2. **Product relationships**

This makes graph-based approaches particularly suitable for recommendation systems.

---

# 🏗️ System Architecture

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    E-COMMERCE PLATFORM                      │
│                                                             │
│  Users │ Products │ Search │ Cart │ Wishlist │ Purchases   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Event Collection   │
                 │                      │
                 │ View / Click / Cart  │
                 │ Wishlist / Purchase  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Graph Construction │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       GNN Model      │
                 │                      │
                 │ LightGCN / GraphSAGE │
                 │       / GAT          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ User/Product         │
                 │ Embeddings           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Candidate Generation │
                 │                      │
                 │ Similarity / FAISS   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Filtering       │
                 │                      │
                 │ Stock / Purchase /   │
                 │ Business Constraints │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       Ranking        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Top-N           │
                 │ Recommendations      │
                 └──────────┬───────────┘
                            │
                            ▼
                    Recommendation API
```

---

# 🔗 Graph Architecture

The initial graph consists of **User** and **Product** nodes.

Additional entities can later be introduced to create a heterogeneous graph.

## Node Types

| Node       | Description                          |
| ---------- | ------------------------------------ |
| `User`     | Customer using the platform          |
| `Product`  | Product available for recommendation |
| `Category` | Product category                     |
| `Brand`    | Product manufacturer/brand           |
| `Session`  | Optional temporary user session      |

---

## Edge Types

| Edge           | Description                 | Example            |
| -------------- | --------------------------- | ------------------ |
| `VIEWED`       | User viewed product         | User → Product     |
| `CLICKED`      | User clicked product        | User → Product     |
| `CART`         | Product added to cart       | User → Product     |
| `WISHLISTED`   | Product added to wishlist   | User → Product     |
| `PURCHASED`    | Product purchased           | User → Product     |
| `BELONGS_TO`   | Product belongs to category | Product → Category |
| `HAS_BRAND`    | Product belongs to brand    | Product → Brand    |
| `CO_VIEWED`    | Products viewed together    | Product ↔ Product  |
| `CO_PURCHASED` | Products purchased together | Product ↔ Product  |

---

# 📊 Interaction Weighting

Not all interactions have equal importance.

A possible initial weighting scheme is:

```text
VIEWED       → 1
CLICKED      → 2
WISHLISTED   → 3
CART         → 4
PURCHASED    → 5
```

These values are experimental and should be tuned through validation.

The model can also incorporate **recency weighting** so that recent interactions have greater influence.

For example:

```text
weight = interaction_weight × exp(-λ × time_since_interaction)
```

---

# 🧮 Recommendation Engine

The recommendation engine is divided into seven major stages.

```text
1. Feature Processing
        ↓
2. Graph Construction
        ↓
3. GNN Representation Learning
        ↓
4. Candidate Generation
        ↓
5. Candidate Filtering
        ↓
6. Ranking
        ↓
7. Post-Processing
```

---

# 1️⃣ Feature Processing

Feature processing converts raw e-commerce data into model-ready representations.

## User Features

Examples:

* Number of views
* Number of purchases
* Number of cart additions
* Preferred categories
* Preferred brands
* Average order value
* Purchase frequency
* Recent activity
* Session activity

## Product Features

Examples:

* Category
* Brand
* Price
* Discount
* Inventory
* Popularity
* Number of views
* Number of purchases
* Product description embedding
* Product image embedding

## Interaction Features

Examples:

* Interaction type
* Timestamp
* Recency
* Frequency
* Session ID
* Purchase quantity
* Transaction value

---

# 2️⃣ Graph Construction

The processed data is transformed into a graph representation compatible with PyTorch Geometric.

Example:

```text
User U1
   │
   ├── VIEWED ──────► Product P1
   │
   ├── CLICKED ─────► Product P2
   │
   ├── CART ────────► Product P3
   │
   └── PURCHASED ───► Product P4
```

The graph builder is responsible for:

* Creating node IDs
* Creating edge indices
* Creating edge attributes
* Creating node features
* Mapping categorical IDs
* Removing invalid interactions
* Creating graph snapshots
* Saving processed graph data

---

# 3️⃣ GNN Model

The initial implementation supports:

* **LightGCN**
* **GraphSAGE**
* **GAT**

## Recommended Initial Model

### LightGCN

LightGCN is the preferred first model because the initial recommendation problem is primarily collaborative filtering between users and products.

Conceptually:

```text
User/Product Initial Embeddings
              │
              ▼
        Graph Propagation
              │
              ▼
        Layer 1 Embeddings
              │
              ▼
        Layer 2 Embeddings
              │
              ▼
     Final Node Representations
```

The final embeddings represent the user's collaborative preferences and the product's position in the interaction graph.

---

# 4️⃣ User-Item Scoring

After obtaining embeddings:

```text
User Embedding
      │
      │
      ▼
   Dot Product
      ▲
      │
      │
Product Embedding
```

The recommendation score can initially be calculated as:

```text
score(u, i) = embedding(u) · embedding(i)
```

A higher score indicates stronger predicted preference.

---

# 5️⃣ Training

The system primarily uses **implicit feedback**.

Instead of asking:

> "How many stars did the user give the product?"

the system learns from actions such as:

```text
VIEW
CLICK
CART
WISHLIST
PURCHASE
```

## Negative Sampling

For every positive interaction:

```text
User U1
   │
   ├── PURCHASED → Product P1   ← Positive
   │
   └── UNKNOWN   → Product P8   ← Negative
```

Negative samples are selected from products the user has not interacted with during the relevant training period.

---

# 6️⃣ Ranking Objective

A suitable initial objective is the **Bayesian Personalized Ranking (BPR)** loss.

```text
L = -log σ(score(u, i+) - score(u, i-))
```

Where:

* `u` = user
* `i+` = positive product
* `i-` = negative product
* `σ` = sigmoid function

The objective encourages:

```text
score(user, positive product)
        >
score(user, negative product)
```

---

# 7️⃣ Candidate Generation

Scoring every product in a large catalog can be expensive.

Therefore, the engine uses a two-stage architecture.

```text
                    User
                     │
                     ▼
               User Embedding
                     │
                     ▼
             Candidate Retrieval
                     │
                     ▼
               Top 500 Items
                     │
                     ▼
                  Filtering
                     │
                     ▼
                  Ranking
                     │
                     ▼
                  Top 20
```

Candidate retrieval can initially use vector similarity.

For larger catalogs, **FAISS** can be used for approximate nearest-neighbor search.

---

# 8️⃣ Candidate Filtering

Candidates are filtered before final ranking.

Typical rules include:

* Remove out-of-stock products.
* Remove unavailable products.
* Remove invalid products.
* Remove products already purchased where appropriate.
* Apply regional availability.
* Apply category restrictions.
* Apply business constraints.

Example:

```text
500 Candidates
      │
      ▼
Inventory Filter
      │
      ▼
Purchase Filter
      │
      ▼
Business Rules
      │
      ▼
320 Valid Candidates
```

---

# 9️⃣ Ranking

The ranking layer determines the final order of recommendations.

An initial scoring function could be:

```text
Final Score =
    α × GNN Score
  + β × Popularity
  + γ × Recency
  + δ × User Preference
  + ε × Business Score
```

The weights are configurable and should be optimized through experiments.

A dedicated Learning-to-Rank model can replace this weighted formula in a later version.

---

# 🔟 Post-Processing

The final recommendations should not simply contain the highest-scoring products.

Post-processing can improve:

* Diversity
* Category coverage
* Brand coverage
* Exploration
* Business constraints

Example:

```text
BAD:

Running Shoe
Running Shoe
Running Shoe
Running Shoe
Running Shoe


BETTER:

Running Shoe
Sports T-Shirt
Fitness Watch
Running Bag
Running Shoe
```

---

# 🧊 Cold Start

Cold-start users have no interaction history.

The GNN cannot generate a meaningful personalized embedding when there is no information about the user.

Therefore, fallback strategies are required.

## New User

```text
New User
   │
   ▼
Trending Products
   +
Popular Products
   +
Session Behavior
```

## New Product

Use:

* Category
* Brand
* Price
* Text embedding
* Image embedding
* Initial popularity

The recommendation engine should gradually transition from fallback recommendations to GNN-based personalization as interaction data becomes available.

---

# 📅 Temporal Data Splitting

Randomly splitting recommendation interactions can cause **data leakage**.

The preferred approach is a temporal split.

Example:

```text
January ───────── September
          TRAIN

October
          VALIDATION

November
          TEST
```

The model must only use information available before the prediction period.

This better simulates real-world recommendation behavior.

---

# 📈 Evaluation

The GNN must be compared against simpler baselines.

## Baselines

```text
1. Popularity-based
        ↓
2. Item similarity
        ↓
3. Collaborative Filtering
        ↓
4. LightGCN / GraphSAGE
        ↓
5. GNN + Ranking
```

A complex model is only useful if it improves recommendation quality over reasonable baselines.

---

# 📊 Offline Metrics

The project evaluates:

### Recall@K

Measures how many relevant items appear in the Top-K recommendations.

### Precision@K

Measures how many recommended items are relevant.

### NDCG@K

Measures ranking quality while giving higher importance to relevant items near the top.

### Hit Rate@K

Measures the percentage of users for whom at least one relevant item appears in Top-K.

### Coverage

Measures how much of the product catalog is being recommended.

### Diversity

Measures how different the recommended products are from each other.

---

# 🧪 Online Evaluation

When integrated with a live platform, the engine can be evaluated using:

* Click-through rate
* Add-to-cart rate
* Purchase conversion
* Revenue per session
* Average order value
* Recommendation engagement
* User retention

---

# 🔬 A/B Testing

The recommendation engine can be evaluated using controlled experiments.

```text
                     Users
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          CONTROL            TREATMENT
              │                 │
              ▼                 ▼
        Existing Model       GNN Model
              │                 │
              └────────┬────────┘
                       ▼
                Metric Analysis
```

Example:

```text
Control:
Popularity Recommendation

Treatment:
GNN Recommendation
```

The primary metric should be defined before the experiment begins.

---

# 🗂️ Project Structure

```text
ecommerce-recommendation/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── frontend/
│
├── backend/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── graph/
│
├── recommendation_engine/
│   │
│   ├── config/
│   │
│   ├── preprocessing/
│   │
│   ├── graph/
│   │
│   ├── models/
│   │
│   ├── training/
│   │
│   ├── embeddings/
│   │
│   ├── candidate_generation/
│   │
│   ├── filtering/
│   │
│   ├── ranking/
│   │
│   ├── inference/
│   │
│   ├── evaluation/
│   │
│   └── utils/
│
├── api/
│
├── experiments/
│   ├── notebooks/
│   └── configs/
│
├── models/
│   ├── checkpoints/
│   └── production/
│
├── cache/
│
├── tests/
│
├── scripts/
│
└── deployment/
```

---

# 👥 Team Responsibilities

The recommendation engine is divided between four team members.

## Person 1 — Data & Graph Engineer

Responsible for:

```text
recommendation_engine/
├── preprocessing/
└── graph/
```

### Responsibilities

* Data cleaning
* Feature engineering
* Interaction weighting
* Temporal splitting
* Graph construction
* Node mapping
* Edge generation
* PyTorch Geometric dataset creation

### Primary Technologies

* Python
* Pandas
* NumPy
* PostgreSQL
* PyTorch Geometric

---

## Person 2 — GNN / ML Engineer

Responsible for:

```text
recommendation_engine/
├── models/
├── training/
└── embeddings/
```

### Responsibilities

* LightGCN
* GraphSAGE
* GAT experiments
* Negative sampling
* BPR loss
* Training pipeline
* Hyperparameter tuning
* Model checkpoints
* Embedding generation

### Primary Technologies

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* scikit-learn
* Matplotlib

---

## Person 3 — Candidate & Ranking Engineer

Responsible for:

```text
recommendation_engine/
├── candidate_generation/
├── filtering/
└── ranking/
```

### Responsibilities

* Candidate retrieval
* Vector similarity
* FAISS integration
* Filtering
* Ranking
* Diversity
* Business rules
* Top-K recommendation generation

### Primary Technologies

* Python
* PyTorch
* NumPy
* FAISS
* scikit-learn

---

## Person 4 — Integration & Evaluation Engineer

Responsible for:

```text
recommendation_engine/
├── inference/
└── evaluation/

api/
```

### Responsibilities

* Recommendation inference
* Engine integration
* Recommendation API
* Evaluation framework
* Metric calculation
* Baseline comparison
* Testing
* Model integration
* Performance monitoring

### Primary Technologies

* Python
* FastAPI
* Pydantic
* Redis
* Pytest
* MLflow

---

# 🛠️ Technology Stack

| Component             | Technology               |
| --------------------- | ------------------------ |
| Programming Language  | Python                   |
| Deep Learning         | PyTorch                  |
| Graph Neural Networks | PyTorch Geometric        |
| GNN Models            | LightGCN, GraphSAGE, GAT |
| Data Processing       | Pandas, NumPy            |
| Database              | PostgreSQL               |
| Vector Search         | FAISS                    |
| Backend API           | FastAPI                  |
| Caching               | Redis                    |
| Experiment Tracking   | MLflow                   |
| Testing               | Pytest                   |
| Containerization      | Docker                   |
| Frontend              | React / Next.js          |
| Visualization         | Matplotlib               |
| Version Control       | Git + GitHub             |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ecommerce-recommendation.git

cd ecommerce-recommendation
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Create a `.env` file based on `.env.example`.

Example:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/recommendation_db

REDIS_URL=redis://localhost:6379

MODEL_PATH=models/production/current_model.pt

EMBEDDING_PATH=recommendation_engine/embeddings/

ENVIRONMENT=development
```

---

# 🏃 Running the Pipeline

## Step 1 — Preprocess data

```bash
python scripts/preprocess.py
```

## Step 2 — Build graph

```bash
python scripts/build_graph.py
```

## Step 3 — Train model

```bash
python scripts/train_model.py
```

## Step 4 — Generate embeddings

```bash
python scripts/generate_embeddings.py
```

## Step 5 — Build FAISS index

```bash
python scripts/build_faiss_index.py
```

## Step 6 — Evaluate model

```bash
python scripts/evaluate_model.py
```

## Step 7 — Start recommendation API

```bash
uvicorn api.main:app --reload
```

The API will be available locally through the configured FastAPI server.

---

# 🔌 Recommendation API

## Get Recommendations

```http
GET /recommendations/{user_id}?k=20
```

Example:

```http
GET /recommendations/U101?k=20
```

Example response:

```json
{
    "user_id": "U101",
    "model_version": "lightgcn_v1",
    "recommendations": [
        {
            "product_id": "P501",
            "score": 0.932
        },
        {
            "product_id": "P103",
            "score": 0.891
        },
        {
            "product_id": "P821",
            "score": 0.874
        }
    ]
}
```

---

# 🧩 Core Engine Interface

The recommendation engine should expose a simple interface:

```python
recommendations = recommender.recommend(
    user_id="U101",
    k=20
)
```

Internally:

```text
recommend()
    │
    ├── Load user embedding
    │
    ├── Generate candidates
    │
    ├── Apply filters
    │
    ├── Calculate ranking scores
    │
    ├── Apply diversity
    │
    └── Return Top-K
```

The engine should remain independent from the frontend and application-specific UI logic.

---

# 🧪 Testing

Run all tests using:

```bash
pytest
```

Test categories include:

```text
Data preprocessing
Graph construction
GNN model
Training
Candidate generation
Filtering
Ranking
Recommendation inference
API
```

Example:

```bash
pytest tests/test_gnn.py
```

---

# 📦 Model Versioning

Models should be versioned rather than overwritten.

Example:

```text
models/
├── checkpoints/
│   ├── lightgcn_v1.pt
│   ├── lightgcn_v2.pt
│   └── lightgcn_v3.pt
│
└── production/
    └── current_model.pt
```

Each production model should have associated metadata:

```text
Model version
Training dataset version
Graph version
Hyperparameters
Training date
Evaluation metrics
```

---

# 🔄 Model Training Pipeline

```text
Raw Interaction Data
        │
        ▼
Data Validation
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Graph Construction
        │
        ▼
Train / Validation / Test Split
        │
        ▼
GNN Training
        │
        ▼
Model Evaluation
        │
        ├──── Worse → Reject
        │
        ▼
Better
        │
        ▼
Model Registry
        │
        ▼
Embedding Generation
        │
        ▼
FAISS Index
        │
        ▼
Production
```

---

# 📈 Production Recommendation Pipeline

```text
User Request
     │
     ▼
Recommendation API
     │
     ▼
User Context
     │
     ▼
User Embedding
     │
     ▼
Candidate Retrieval
     │
     ▼
Top 500
     │
     ▼
Filtering
     │
     ▼
Ranking
     │
     ▼
Diversity / Business Rules
     │
     ▼
Top 20
     │
     ▼
API Response
```

---

# ⚡ Performance Considerations

The engine is designed around a two-stage recommendation architecture.

### Stage 1 — Retrieval

Fast and broad:

```text
Millions of products
        ↓
Vector Retrieval
        ↓
Hundreds of candidates
```

### Stage 2 — Ranking

More expensive but applied to a much smaller set:

```text
Hundreds of candidates
        ↓
Feature-based ranking
        ↓
Top 20
```

This architecture allows the system to scale much better than scoring the entire catalog for every request.

---

# 🔐 Security Considerations

The recommendation engine should:

* Never expose private user interaction history.
* Validate incoming user IDs.
* Authenticate API requests.
* Rate-limit recommendation endpoints.
* Protect database credentials.
* Store secrets in environment variables or a secret manager.
* Avoid logging sensitive user information.
* Validate event payloads.
* Maintain model versioning for reproducibility.

---

# 📊 Monitoring

Production monitoring should track:

### System Metrics

* API latency
* API error rate
* CPU usage
* GPU usage
* Memory usage
* Recommendation cache hit rate

### Data Metrics

* Number of interactions
* Missing events
* New users
* New products
* Graph size
* Embedding freshness

### Recommendation Metrics

* Recall@K
* NDCG@K
* Coverage
* Diversity
* CTR
* Conversion rate
* Revenue per recommendation

---

# 🗺️ Development Roadmap

## Phase 1 — Foundation

* [ ] Set up repository
* [ ] Define database schema
* [ ] Collect interaction data
* [ ] Implement event logging
* [ ] Create preprocessing pipeline

## Phase 2 — Graph

* [ ] Build User–Product graph
* [ ] Implement node mapping
* [ ] Implement edge construction
* [ ] Add interaction weights
* [ ] Add temporal splitting

## Phase 3 — Baselines

* [ ] Popularity recommender
* [ ] Item similarity
* [ ] Collaborative filtering
* [ ] Establish baseline metrics

## Phase 4 — GNN

* [ ] Implement LightGCN
* [ ] Implement negative sampling
* [ ] Implement BPR loss
* [ ] Train first model
* [ ] Generate embeddings

## Phase 5 — Retrieval

* [ ] Implement similarity search
* [ ] Implement candidate generation
* [ ] Integrate FAISS
* [ ] Implement filtering

## Phase 6 — Ranking

* [ ] Implement ranking features
* [ ] Implement initial ranking formula
* [ ] Add diversity
* [ ] Compare ranking strategies

## Phase 7 — Serving

* [ ] Build inference module
* [ ] Create FastAPI endpoint
* [ ] Add Redis caching
* [ ] Add model loading

## Phase 8 — Evaluation

* [ ] Implement Recall@K
* [ ] Implement Precision@K
* [ ] Implement NDCG@K
* [ ] Implement Hit Rate@K
* [ ] Implement Coverage
* [ ] Implement Diversity
* [ ] Compare against baselines

## Phase 9 — Advanced GNN

* [ ] Add Category nodes
* [ ] Add Brand nodes
* [ ] Experiment with GraphSAGE
* [ ] Experiment with GAT
* [ ] Investigate heterogeneous GNNs

## Phase 10 — Production

* [ ] Dockerize services
* [ ] Add MLflow
* [ ] Add monitoring
* [ ] Implement model versioning
* [ ] Implement automated retraining
* [ ] Deploy recommendation engine

---

# 🔮 Future Improvements

Potential extensions include:

### Heterogeneous Graph

```text
User
 │
 ├──► Product
 │
 ├──► Category
 │
 └──► Brand
```

### Session-Based Recommendation

Incorporate the user's current session:

```text
User History
     +
Current Session
     ↓
Recommendation
```

### Multimodal Recommendation

Combine:

```text
Text Embedding
      +
Image Embedding
      +
Graph Embedding
```

### Temporal GNN

Model how preferences change over time.

### Learning-to-Rank

Replace manually weighted ranking with a trainable ranking model.

### Real-Time Personalization

Update recommendations based on recent interactions without waiting for a complete model retraining cycle.

---

# 🏆 Expected Final System

The final recommendation engine should provide:

```text
                 ┌─────────────────────┐
                 │      User U101      │
                 └──────────┬──────────┘
                            │
                            ▼
                    User Embedding
                            │
                            ▼
                  Candidate Retrieval
                            │
                            ▼
                       500 Items
                            │
                            ▼
                       Filtering
                            │
                            ▼
                       300 Items
                            │
                            ▼
                         Ranking
                            │
                            ▼
                       Diversity
                            │
                            ▼
                    ┌───────────────┐
                    │   TOP 20      │
                    │ Recommendations│
                    └───────────────┘
```

The complete system therefore follows:

```text
DATA
  ↓
GRAPH
  ↓
GNN
  ↓
EMBEDDINGS
  ↓
RETRIEVAL
  ↓
FILTERING
  ↓
RANKING
  ↓
DIVERSITY
  ↓
TOP-N
  ↓
API
  ↓
E-COMMERCE PLATFORM
```

---

# 👨‍💻 Team

| Member       | Responsibility                       |
| ------------ | ------------------------------------ |
| **Member 1** | Data Processing & Graph Construction |
| **Member 2** | GNN Architecture & Model Training    |
| **Member 3** | Candidate Generation & Ranking       |
| **Member 4** | Integration, API & Evaluation        |

---

# 📄 Project Philosophy

This project is designed around a simple principle:

> **The GNN is not the entire recommendation system. It is the representation-learning component inside a larger recommendation engine.**

A strong recommendation system requires:

```text
Clean Data
+
Correct Graph Representation
+
Good GNN
+
Efficient Retrieval
+
Effective Ranking
+
Business Constraints
+
Evaluation
+
Reliable Serving
```

The goal of this project is therefore not merely to train a GNN, but to build a **complete, measurable, scalable recommendation pipeline**.

---

# ⭐ Project Status

**Status:** 🚧 In Development

Current focus:

```text
[ ] Data Pipeline
[ ] Graph Construction
[ ] Baseline Recommendation
[ ] LightGCN
[ ] Candidate Generation
[ ] Ranking
[ ] Evaluation
[ ] Recommendation API
[ ] Deployment
```

---

# 📜 License

This project is intended for educational and research purposes.

A formal open-source license can be added before public distribution.
