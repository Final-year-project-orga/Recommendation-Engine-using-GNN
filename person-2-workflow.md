# LightGCN Recommendation Pipeline

## Step 1: The Graph & The Normalized Adjacency Matrix

In **`matrix_create.py`**:

- A user-item graph is **bipartite** (Users only connect to Products, never directly to Users).
- We combine users and items into a single global index:
  - Users `0…5` → indices `0…5`
  - Products `0…9` → indices `6…15` (offset by `num_users`)
  - Total nodes = `6 + 10 = 16`.
- If we just multiplied by the raw adjacency matrix `A`, popular products with 1,000 views would explode in magnitude, while niche products would shrink to 0.
- Therefore, we compute the **symmetric normalized adjacency matrix**:

  `Ã = D⁻¹/² A D⁻¹/²`

  where `D` is the degree matrix (how many connections each node has).

This normalizes the message passing so high-degree nodes don't blow up the gradients.

---

## Step 2: Why LightGCN? (The Core Model)

In **`lightgcn.py`** and **`layers.py`**:

- Traditional GCN uses feature projection matrices (`W`) and non-linear activations (`ReLU`):

  `h⁽ᵏ⁺¹⁾ = σ(W Ã h⁽ᵏ⁾)`

- **The LightGCN Insight (He et al., 2020):** In recommendation systems, nodes don't have rich continuous input features like image pixels. They only have ID embeddings.
  - Adding weights `W` and `ReLU` actually **harms** collaborative filtering performance and makes training slower.
  - LightGCN removes `W` and `ReLU`, keeping **only pure linear neighborhood propagation**:

  `E⁽ᵏ⁺¹⁾ = Ã · E⁽ᵏ⁾`

### What does each hop mean?

- **Layer 0 (`E⁽⁰⁾`)**: The node's raw initial embedding (self preference).
- **Layer 1 (`E⁽¹⁾`)**: User aggregates the products they interacted with. Product aggregates users who bought them.
- **Layer 2 (`E⁽²⁾`)**: 2 hops away! A user aggregates other users with similar tastes (Collaborative Filtering).
- **Layer 3 (`E⁽³⁾`)**: 3 hops away! Higher-order graph structural context.

### Combining Layers

To avoid **over-smoothing** (where all nodes look identical if propagated too many times), LightGCN averages the embeddings across all layers:

`E = 1/(K+1) Σₖ₌₀ᴷ E⁽ᵏ⁾`

---

## Step 3: Implicit Feedback & Negative Sampling

In **`negative_sampling.py`**:

- In e-commerce, users rarely leave 1-to-5 star ratings. We only see **implicit feedback** (clicks, carts, purchases).
- A missing interaction does **not** necessarily mean the user hates the product—it just means they haven't seen it yet.
- For every positive pair `(u, i)` (e.g., User 0 viewed Product 1), our `NegativeSampler` picks a random product `j` that User 0 has **never** interacted with (e.g., Product 8).
- This produces training triplets:

  `(user u, positive product i, negative product j)`

---

## Step 4: Bayesian Personalized Ranking (BPR) Loss

In **`losses.py`** & **`trainer.py`**:

- The model computes predicted preference scores using the dot product:

  `ŷᵘⁱ = eᵤᵀ eᵢ` — score for positive item

  `ŷᵘʲ = eᵤᵀ eⱼ` — score for negative item

- The BPR objective aims to ensure:

  `ŷᵘⁱ > ŷᵘʲ`

- The loss function is:

  `L_BPR = −ln σ(ŷᵘⁱ − ŷᵘʲ) + λ/2 (||eᵤ⁽⁰⁾||² + ||eᵢ⁽⁰⁾||² + ||eⱼ⁽⁰⁾||²)`

- If the model predicts `ŷᵘⁱ ≫ ŷᵘʲ`, the difference is large positive, `σ(diff) ≈ 1`, and `−ln(1) = 0` (loss is near zero).
- If the model predicts `ŷᵘⁱ < ŷᵘʲ`, the loss is large, pushing gradients back through PyTorch to update `E⁽⁰⁾` so that user `u` moves closer to positive item `i` and farther from negative item `j`.

---

## Step 5 & 6: Checkpointing & Exporting Embeddings

In **`generate_embeddings.py`**:

- When training finishes, the best weights are saved in:

  `models/checkpoints/lightgcn_best.pt`

- `generate_embeddings.py` loads the model, switches to `.eval()` mode, does a single forward pass over the graph, and extracts:
  - `user_embeddings.npy` — matrix of shape `(6, 64)`
  - `product_embeddings.npy` — matrix of shape `(10, 64)`

These exported embeddings can then be used by the recommendation system to calculate user-product similarity and generate personalized recommendations.
