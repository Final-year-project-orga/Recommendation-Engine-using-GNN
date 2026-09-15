"""
Unit tests for Person 2 (GNN / ML Engineer) components:
- LightGCNLayer
- LightGCN
- GraphSAGE
- GAT
- BPRLoss
- NegativeSampler
- CheckpointManager
- Trainer & Training loop
- Embedding generation & loading
"""
import os
import shutil
import tempfile
import unittest
import torch

from reccomendation_engine.convertor import num_users, num_items, interactions
from reccomendation_engine.models.matrix_create import make_n_adj_mat, make_adj_mat
from reccomendation_engine.models.layers import LightGCNLayer
from reccomendation_engine.models.lightgcn import LightGCN
from reccomendation_engine.models.graphsage import GraphSAGE
from reccomendation_engine.models.gat import GAT
from reccomendation_engine.training.losses import BPRLoss, PointwiseBCELoss
from reccomendation_engine.training.negative_sampling import NegativeSampler
from reccomendation_engine.training.checkpoint import CheckpointManager
from reccomendation_engine.training.trainer import Trainer
from reccomendation_engine.embeddings.generate_embeddings import (
    generate_and_save_embeddings,
    load_embeddings,
)


class TestGNNComponents(unittest.TestCase):
    def setUp(self):
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items
        self.embedding_dim = 16
        self.n_adj_np = make_n_adj_mat()
        self.n_adj = torch.tensor(self.n_adj_np, dtype=torch.float32)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_lightgcn_layer_dense_and_sparse(self):
        layer = LightGCNLayer()
        x = torch.randn(self.total_nodes, self.embedding_dim)

        # Dense
        out_dense = layer(self.n_adj, x)
        self.assertEqual(out_dense.shape, (self.total_nodes, self.embedding_dim))

        # Sparse
        sparse_adj = self.n_adj.to_sparse()
        out_sparse = layer(sparse_adj, x)
        self.assertEqual(out_sparse.shape, (self.total_nodes, self.embedding_dim))
        self.assertTrue(torch.allclose(out_dense, out_sparse, atol=1e-5))

    def test_lightgcn_model_forward_and_predict(self):
        model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=3,
        )
        u_emb, i_emb, layers_emb = model(self.n_adj)
        self.assertEqual(u_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(i_emb.shape, (self.num_items, self.embedding_dim))
        self.assertEqual(len(layers_emb), 4)  # Layer 0, 1, 2, 3

        # Test predict
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 2, 4])
        scores = model.predict(users, items, self.n_adj)
        self.assertEqual(scores.shape, (3,))

    def test_graphsage_and_gat(self):
        sage = GraphSAGE(self.num_users, self.num_items, embedding_dim=self.embedding_dim, num_layers=2)
        u_sage, i_sage = sage(self.n_adj)
        self.assertEqual(u_sage.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(i_sage.shape, (self.num_items, self.embedding_dim))

        raw_adj = torch.tensor(make_adj_mat(), dtype=torch.float32)
        gat = GAT(self.num_users, self.num_items, embedding_dim=self.embedding_dim, num_heads=2, num_layers=2)
        u_gat, i_gat = gat(raw_adj)
        self.assertEqual(u_gat.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(i_gat.shape, (self.num_items, self.embedding_dim))

    def test_negative_sampler(self):
        sampler = NegativeSampler(self.num_users, self.num_items, interactions)
        for _ in range(50):
            for u, pos_items in sampler.user_pos_items.items():
                if len(pos_items) < self.num_items:
                    neg = sampler.sample_negative_item_for_user(u)
                    self.assertNotIn(neg, pos_items, f"Sampled item {neg} is actually in user {u}'s positive set!")

        users, pos_items, neg_items = sampler.sample_epoch_triplets()
        self.assertEqual(len(users), len(interactions))
        self.assertEqual(len(pos_items), len(interactions))
        self.assertEqual(len(neg_items), len(interactions))

    def test_bpr_loss(self):
        loss_fn = BPRLoss(reg_weight=1e-3)
        # When pos_scores > neg_scores, loss should be small
        pos_high = torch.tensor([5.0, 4.0, 6.0])
        neg_low = torch.tensor([0.1, -1.0, 0.0])
        loss_good, bpr_good, _ = loss_fn(pos_high, neg_low)

        # When pos_scores < neg_scores, loss should be large
        pos_low = torch.tensor([0.1, -1.0, 0.0])
        neg_high = torch.tensor([5.0, 4.0, 6.0])
        loss_bad, bpr_bad, _ = loss_fn(pos_low, neg_high)

        self.assertLess(loss_good.item(), loss_bad.item())

    def test_checkpoint_manager(self):
        mgr = CheckpointManager(save_dir=self.test_dir)
        model = LightGCN(self.num_users, self.num_items, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        saved_path = mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.42,
            filename="test_ckpt.pt",
        )
        self.assertTrue(os.path.exists(saved_path))

        new_model = LightGCN(self.num_users, self.num_items, embedding_dim=self.embedding_dim)
        data = mgr.load_checkpoint(saved_path, new_model)
        self.assertEqual(data["epoch"], 5)
        self.assertAlmostEqual(data["loss"], 0.42)

        # Check weights are identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_trainer_and_embeddings_pipeline(self):
        model = LightGCN(self.num_users, self.num_items, embedding_dim=16, num_layers=2)
        sampler = NegativeSampler(self.num_users, self.num_items, interactions)
        ckpt_mgr = CheckpointManager(save_dir=self.test_dir)
        trainer = Trainer(
            model=model,
            n_adj=self.n_adj,
            sampler=sampler,
            lr=0.05,
            checkpoint_manager=ckpt_mgr,
        )

        history = trainer.fit(epochs=10, print_every=5, save_best=True, model_name="test_model.pt")
        self.assertEqual(len(history["total_loss"]), 10)
        # Loss should decrease during training
        self.assertLess(history["total_loss"][-1], history["total_loss"][0])

        # Test embedding generation and loading
        u_emb, i_emb = generate_and_save_embeddings(
            model=model,
            save_dir=self.test_dir,
            save_numpy=True,
        )
        self.assertEqual(u_emb.shape, (self.num_users, 16))
        self.assertEqual(i_emb.shape, (self.num_items, 16))

        loaded_u, loaded_i, meta = load_embeddings(embedding_dir=self.test_dir, as_numpy=True)
        self.assertEqual(loaded_u.shape, (self.num_users, 16))
        self.assertEqual(loaded_i.shape, (self.num_items, 16))
        self.assertEqual(meta["num_users"], self.num_users)


if __name__ == "__main__":
    unittest.main()
