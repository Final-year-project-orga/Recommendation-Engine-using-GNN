from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    r"""
    Bayesian Personalized Ranking (BPR) Loss with L2 Regularization.
    Reference: Rendle et al., UAI 2009 & He et al., SIGIR 2020 (LightGCN).

    L_BPR = - \sum_{(u, i, j)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|^2
    """
    def __init__(self, reg_weight: float = 1e-4):
        r"""
        Args:
            reg_weight: L2 regularization coefficient (\lambda)
        """
        super().__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        user_ego_emb: Optional[torch.Tensor] = None,
        pos_item_ego_emb: Optional[torch.Tensor] = None,
        neg_item_ego_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pos_scores: (batch_size,) predicted score for positive pairs (u, i)
            neg_scores: (batch_size,) predicted score for negative pairs (u, j)
            user_ego_emb: (batch_size, dim) initial ego embeddings of users (Layer 0)
            pos_item_ego_emb: (batch_size, dim) initial ego embeddings of positive items (Layer 0)
            neg_item_ego_emb: (batch_size, dim) initial ego embeddings of negative items (Layer 0)

        Returns:
            loss: total loss (bpr_loss + reg_loss)
            bpr_loss: mean BPR ranking loss
            reg_loss: mean L2 regularization loss
        """
        # Numerically stable: -log(sigmoid(x)) = softplus(-x)
        diff = pos_scores - neg_scores
        bpr_loss = F.softplus(-diff).mean()

        reg_loss = torch.tensor(0.0, device=pos_scores.device)
        if self.reg_weight > 0.0 and user_ego_emb is not None:
            # L2 norm on initial embeddings (0-th layer) as in LightGCN paper
            user_reg = (user_ego_emb ** 2).sum(dim=-1).mean()
            pos_reg = (pos_item_ego_emb ** 2).sum(dim=-1).mean() if pos_item_ego_emb is not None else 0.0
            neg_reg = (neg_item_ego_emb ** 2).sum(dim=-1).mean() if neg_item_ego_emb is not None else 0.0

            reg_loss = (self.reg_weight / 2.0) * (user_reg + pos_reg + neg_reg)

        total_loss = bpr_loss + reg_loss
        return total_loss, bpr_loss, reg_loss


class PointwiseBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for recommendation (pointwise baseline).
    """
    def __init__(self, reg_weight: float = 1e-4):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(scores, labels)
        if self.reg_weight > 0.0 and embeddings is not None:
            reg = self.reg_weight * (embeddings ** 2).sum(dim=-1).mean()
            return bce + reg
        return bce
