```
ecommerce-recommendation/
│
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── docker-compose.yml
│
├── frontend/                         # E-commerce website
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── layouts/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── utils/
│   └── package.json
│
│
├── backend/                          # Main application backend
│   ├── main.py
│   │
│   ├── routes/
│   │   ├── auth.py
│   │   ├── products.py
│   │   ├── users.py
│   │   ├── orders.py
│   │   ├── events.py
│   │   └── recommendations.py
│   │
│   ├── models/
│   │   ├── user.py
│   │   ├── product.py
│   │   ├── order.py
│   │   └── event.py
│   │
│   ├── schemas/
│   │   ├── user.py
│   │   ├── product.py
│   │   ├── order.py
│   │   └── recommendation.py
│   │
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── product_service.py
│   │   ├── order_service.py
│   │   └── recommendation_service.py
│   │
│   └── database/
│       ├── connection.py
│       └── queries.py
│
│
├── data/                             # Data used by recommendation engine
│   ├── raw/
│   │   ├── users.csv
│   │   ├── products.csv
│   │   └── interactions.csv
│   │
│   ├── processed/
│   │   ├── users_processed.csv
│   │   ├── products_processed.csv
│   │   └── interactions_processed.csv
│   │
│   └── graph/
│       ├── nodes/
│       ├── edges/
│       └── graph_data.pt
│
│
├── recommendation_engine/             # ⭐ CORE ML ENGINE
│   │
│   ├── config/
│   │   ├── model_config.yaml
│   │   ├── training_config.yaml
│   │   └── recommendation_config.yaml
│   │
│   ├── preprocessing/                 # PERSON 1
│   │   ├── clean_data.py
│   │   ├── feature_engineering.py
│   │   ├── interaction_weights.py
│   │   └── train_test_split.py
│   │
│   ├── graph/                         # PERSON 1
│   │   ├── graph_builder.py
│   │   ├── node_features.py
│   │   ├── edge_features.py
│   │   ├── graph_schema.py
│   │   └── graph_loader.py
│   │
│   ├── models/                        # PERSON 2
│   │   ├── lightgcn.py
│   │   ├── graphsage.py
│   │   ├── gat.py
│   │   └── layers.py
│   │
│   ├── training/                      # PERSON 2
│   │   ├── train.py
│   │   ├── trainer.py
│   │   ├── negative_sampling.py
│   │   ├── losses.py
│   │   └── checkpoint.py
│   │
│   ├── embeddings/                    # PERSON 2
│   │   ├── generate_embeddings.py
│   │   ├── user_embeddings.pt
│   │   └── product_embeddings.pt
│   │
│   ├── candidate_generation/          # PERSON 3
│   │   ├── candidate_generator.py
│   │   ├── similarity.py
│   │   ├── faiss_index.py
│   │   └── retrieval.py
│   │
│   ├── filtering/                     # PERSON 3
│   │   ├── inventory_filter.py
│   │   ├── purchase_filter.py
│   │   └── business_rules.py
│   │
│   ├── ranking/                       # PERSON 3
│   │   ├── ranker.py
│   │   ├── scoring.py
│   │   ├── features.py
│   │   └── diversity.py
│   │
│   ├── inference/                     # PERSON 4
│   │   ├── inference.py
│   │   ├── recommender.py
│   │   └── model_loader.py
│   │
│   ├── evaluation/                    # PERSON 4
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   ├── recall.py
│   │   ├── ndcg.py
│   │   ├── precision.py
│   │   └── baseline_comparison.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py
│       └── helpers.py
│
│
├── api/                               # Recommendation API
│   ├── main.py
│   ├── routes/
│   │   └── recommendation_routes.py
│   ├── schemas/
│   │   └── recommendation_schema.py
│   └── dependencies.py
│
│
├── experiments/                       # Experiments / notebooks
│   ├── notebooks/
│   │   ├── data_analysis.ipynb
│   │   ├── graph_analysis.ipynb
│   │   ├── gnn_experiment.ipynb
│   │   └── evaluation.ipynb
│   │
│   └── configs/
│       ├── experiment_01.yaml
│       └── experiment_02.yaml
│
│
├── models/                            # Trained model artifacts
│   ├── checkpoints/
│   │   ├── lightgcn_v1.pt
│   │   └── lightgcn_v2.pt
│   │
│   └── production/
│       └── current_model.pt
│
│
├── cache/                             # Local development cache
│   ├── embeddings/
│   └── recommendations/
│
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_graph.py
│   ├── test_gnn.py
│   ├── test_candidate_generation.py
│   ├── test_ranking.py
│   ├── test_recommendations.py
│   └── test_api.py
│
│
├── scripts/
│   ├── build_graph.py
│   ├── train_model.py
│   ├── generate_embeddings.py
│   ├── build_faiss_index.py
│   └── evaluate_model.py
│
│
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── nginx.conf
    └── monitoring/
        ├── prometheus.yml
        └── grafana/
```