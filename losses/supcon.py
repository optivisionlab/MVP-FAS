import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls embeddings of the same class together and pushes different classes
    apart on the unit hypersphere.

    Args:
        temperature: scaling factor for similarity logits.
        base_temperature: kept for compat with the original paper formulation.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: [B, D] tensor, expected L2-normalized by caller.
        labels:   [B]   tensor of integer class ids.
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.zeros([], device=device, dtype=features.dtype, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature

        # logsumexp trick for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # remove self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(dim=1)
        # guard against anchors with no positive in the batch
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6,
            torch.ones_like(mask_pos_pairs),
            mask_pos_pairs,
        )
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()
