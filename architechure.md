```
E-COMMERCE WEBSITE
       │
React / Next.js
       │
       ▼
   FastAPI
┌─────────┴─────────┐
│                   │
Events          Recommendation
│                   │
▼                   ▼
Event Database       Candidate Retrieval
│                   │
▼                   ▼
Graph Construction     GNN Embeddings
│              (GraphSAGE /
│               LightGCN)
│                   │
└───────┬───────────┘
     ▼
Ranking + Filtering
     │
     ▼
Top-N Products
     │
     ▼
Frontend UI
```


# **Engine Architechture**
```
┌─────────────────────────────┐
│       USER CONTEXT          │
│                             │
│ user_id                     │
│ recent interactions         │
│ session behavior            │
│ preferences                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     FEATURE / CONTEXT       │
│        PROCESSOR            │
│                             │
│ • User features             │
│ • Item features             │
│ • Interaction features      │
│ • Recency                   │
└──────────────┬──────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│             GRAPH BUILDER               │
│                                        │
│   User ──VIEW──> Product               │
│   User ──CART──> Product               │
│   User ──BUY───> Product               │
│   Product ──> Category                 │
│   Product ──> Brand                    │
│                                        │
└───────────────────┬────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│        GRAPH DATA           │
│                             │
│ Nodes:                      │
│ User / Product / Brand      │
│ Category                    │
│                             │
│ Edges:                      │
│ View / Click / Cart / Buy   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          GNN MODEL           │
│                             │
│  GraphSAGE / LightGCN / GAT │
│                             │
│   Message Passing           │
│        ↓                    │
│   Node Embeddings           │
│        ↓                    │
│ User Embedding              │
│ Product Embedding           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    CANDIDATE GENERATION     │
│                             │
│ User Embedding              │
│        +                    │
│ Product Embeddings           │
│        ↓                    │
│ Similarity / Dot Product    │
│        ↓                    │
│ Top 100–1000 candidates     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          FILTERING           │
│                             │
│ • Out of stock              │
│ • Already purchased         │
│ • Invalid products          │
│ • User/business constraints │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│           RANKER             │
│                             │
│ GNN Score                   │
│ +                           │
│ User/Product Features       │
│ +                           │
│ Context                     │
│        ↓                    │
│ Final Recommendation Score  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       POST PROCESSING       │
│                             │
│ • Diversity                │
│ • Category balancing        │
│ • Business rules            │
│ • Exploration               │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         TOP-N ITEMS          │
│                             │
│ Product 1                   │
│ Product 2                   │
│ Product 3                   │
│ ...                         │
│ Product 20                  │
└─────────────────────────────┘
```