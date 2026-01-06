"""
Supervised Routing Module for MoE Models

Implements soft supervision of router decisions based on expert labels.
Provides components for guiding MoE routing using dataset-provided expert categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ExpertLabelEmbedding(nn.Module):
    """
    Learnable mapping from dataset expert labels to model expert preferences.

    Maps 4 dataset categories to 8 Mixtral experts using a learned linear projection.
    This allows the model to learn which experts should handle which categories.

    Args:
        num_categories: Number of dataset expert categories (default: 4)
        num_experts: Number of model experts (default: 8)
        temperature: Temperature for softmax scaling (default: 1.0)
    """

    def __init__(
        self,
        num_categories: int = 4,
        num_experts: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.num_experts = num_experts
        self.temperature = temperature

        # Learnable mapping: [num_categories] -> [num_experts]
        # Initialize with slight positive bias to encourage diverse expert usage
        self.label_to_expert_prefs = nn.Linear(num_categories, num_experts)
        nn.init.xavier_uniform_(self.label_to_expert_prefs.weight)
        nn.init.constant_(self.label_to_expert_prefs.bias, 0.1)

        logger.info(
            f"Initialized ExpertLabelEmbedding: {num_categories} categories -> "
            f"{num_experts} experts (temperature={temperature})"
        )

    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert expert label IDs to expert preference logits.

        Args:
            label_ids: [batch_size] - Integer label IDs (0 to num_categories-1)

        Returns:
            expert_prefs: [batch_size, num_experts] - Expert preference logits
        """
        # One-hot encode: [batch_size] -> [batch_size, num_categories]
        one_hot = F.one_hot(label_ids, num_classes=self.num_categories).float()

        # Project to expert preferences: [batch_size, num_categories] -> [batch_size, num_experts]
        expert_prefs = self.label_to_expert_prefs(one_hot)

        # Apply temperature scaling
        expert_prefs = expert_prefs / self.temperature

        return expert_prefs


def compute_routing_supervision_loss(
    router_logits: torch.Tensor,
    expert_prefs: torch.Tensor,
    attention_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence between router decisions and expert label preferences.

    Encourages the router to align its decisions with the expert preferences
    derived from dataset labels, while still allowing learned routing behavior.

    Args:
        router_logits: [batch, seq_len, num_experts] - Router logits from MoE layers
        expert_prefs: [batch, num_experts] - Expert preference logits from labels
        attention_mask: [batch, seq_len] - Mask for valid tokens (1=valid, 0=padding)
        reduction: "mean", "sum", or "none"

    Returns:
        loss: Scalar tensor (if reduction="mean" or "sum") or [batch, seq_len] (if "none")
    """
    batch_size, seq_len, num_experts = router_logits.shape

    # Expand expert preferences to all sequence positions
    # [batch, num_experts] -> [batch, 1, num_experts] -> [batch, seq_len, num_experts]
    expert_prefs_expanded = expert_prefs.unsqueeze(1).expand(-1, seq_len, -1)

    # Convert to probability distributions
    router_probs = F.softmax(router_logits, dim=-1)      # [batch, seq_len, num_experts]
    target_probs = F.softmax(expert_prefs_expanded, dim=-1)  # [batch, seq_len, num_experts]

    # Compute KL divergence: KL(target || router)
    # We want router to match target, so target is P and router is Q
    kl_div = F.kl_div(
        router_probs.log(),
        target_probs,
        reduction='none',
        log_target=False,
    ).sum(dim=-1)  # [batch, seq_len]

    # Mask out padding tokens
    kl_div = kl_div * attention_mask

    if reduction == "mean":
        # Average over valid tokens only
        return kl_div.sum() / attention_mask.sum().clamp(min=1)
    elif reduction == "sum":
        return kl_div.sum()
    else:
        return kl_div


def aggregate_router_logits_across_layers(
    router_logits_tuple: Tuple[torch.Tensor, ...],
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Aggregate router logits from multiple MoE layers.

    Mixtral has 32 decoder layers with MoE in each. We need to aggregate
    routing decisions across all layers for supervision.

    Args:
        router_logits_tuple: Tuple of [batch, seq_len, num_experts] tensors
        aggregation: "mean", "sum", or "last"

    Returns:
        aggregated: [batch, seq_len, num_experts]
    """
    if not router_logits_tuple or len(router_logits_tuple) == 0:
        raise ValueError("router_logits_tuple is empty")

    if aggregation == "last":
        return router_logits_tuple[-1]
    elif aggregation == "mean":
        return torch.stack(router_logits_tuple, dim=0).mean(dim=0)
    elif aggregation == "sum":
        return torch.stack(router_logits_tuple, dim=0).sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


class SupervisedMoEWrapper(nn.Module):
    """
    Wrapper that adds supervised routing capability to pre-trained MoE models.

    Wraps a Mixtral model and adds:
    1. Expert label embedding layer
    2. Routing supervision loss computation
    3. Combined loss calculation

    The wrapper is transparent - it can be used exactly like the underlying model,
    but with the additional capability of using expert labels to guide routing.

    Args:
        moe_model: Pre-trained MoE model (e.g., MixtralForCausalLM)
        num_categories: Number of expert categories in dataset (default: 4)
        num_experts: Number of experts in the model (default: 8)
        routing_loss_weight: Weight for routing supervision loss (default: 0.1)
        label_embedding_temperature: Temperature for expert label embedding (default: 1.0)
        router_aggregation: How to aggregate router logits across layers (default: "mean")
    """

    def __init__(
        self,
        moe_model: nn.Module,
        num_categories: int = 4,
        num_experts: int = 8,
        routing_loss_weight: float = 0.1,
        label_embedding_temperature: float = 1.0,
        router_aggregation: str = "mean",
    ):
        super().__init__()
        self.moe_model = moe_model
        self.num_categories = num_categories
        self.num_experts = num_experts
        self.routing_loss_weight = routing_loss_weight
        self.router_aggregation = router_aggregation

        # Expert label embedding
        self.expert_label_embedding = ExpertLabelEmbedding(
            num_categories=num_categories,
            num_experts=num_experts,
            temperature=label_embedding_temperature,
        )

        logger.info(
            f"Initialized SupervisedMoEWrapper with routing_loss_weight={routing_loss_weight}, "
            f"aggregation={router_aggregation}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        expert_label_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with supervised routing.

        Args:
            input_ids: [batch, seq_len] - Input token IDs
            attention_mask: [batch, seq_len] - Attention mask
            labels: [batch, seq_len] - Target token IDs for language modeling
            expert_label_ids: [batch] - Expert category labels (optional)
            **kwargs: Additional arguments passed to the underlying model

        Returns:
            outputs: Model outputs with modified loss including routing supervision.
                     If expert_label_ids are provided, outputs will include:
                     - loss: Combined loss (LM + load balancing + routing supervision)
                     - routing_supervision_loss: Routing supervision component
        """
        # Forward pass through MoE model
        outputs = self.moe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_router_logits=True,  # CRITICAL: Must capture router logits
            **kwargs,
        )

        # If no expert labels provided or routing supervision disabled, return as-is
        if expert_label_ids is None or self.routing_loss_weight == 0:
            return outputs

        # Extract language modeling loss
        lm_loss = outputs.loss
        if lm_loss is None:
            # If no labels provided, skip loss computation
            return outputs

        # Extract router logits (tuple of tensors, one per MoE layer)
        router_logits = outputs.router_logits
        if router_logits is None or len(router_logits) == 0:
            logger.warning(
                "Router logits not available. Make sure output_router_logits=True "
                "is set in model config."
            )
            return outputs

        try:
            # Compute expert preferences from labels
            expert_prefs = self.expert_label_embedding(expert_label_ids)  # [batch, num_experts]

            # Aggregate router logits across layers
            aggregated_router_logits = aggregate_router_logits_across_layers(
                router_logits,
                aggregation=self.router_aggregation,
            )

            # Compute routing supervision loss
            routing_loss = compute_routing_supervision_loss(
                router_logits=aggregated_router_logits,
                expert_prefs=expert_prefs,
                attention_mask=attention_mask,
            )

            # Combine losses
            # Total loss = LM loss + auxiliary loss (built into model) + routing supervision loss
            total_loss = lm_loss + self.routing_loss_weight * routing_loss

            # Store routing loss for monitoring
            outputs.loss = total_loss
            outputs.routing_supervision_loss = routing_loss

        except Exception as e:
            logger.error(f"Error computing routing supervision loss: {e}")
            # Fall back to original loss if anything goes wrong
            return outputs

        return outputs

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying model.

        This allows the wrapper to be used transparently as if it were the model itself.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.moe_model, name)


def get_expert_utilization_stats(
    router_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    attention_mask: torch.Tensor,
    top_k: int = 2,
) -> dict:
    """
    Compute expert utilization statistics from router logits.

    Useful for monitoring whether all experts are being used effectively
    or if some experts are being neglected (expert collapse).

    Args:
        router_logits: Either single tensor [batch, seq_len, num_experts] or
                      tuple of such tensors from multiple layers
        attention_mask: [batch, seq_len] - Mask for valid tokens
        top_k: Number of experts selected per token (default: 2)

    Returns:
        stats: Dictionary with expert utilization metrics:
            - expert_counts: [num_experts] - Tokens assigned to each expert
            - expert_percentages: [num_experts] - Percentage of tokens per expert
            - utilization_variance: Scalar - Variance in expert usage (lower = more balanced)
            - min_utilization: Scalar - Minimum expert usage percentage
            - max_utilization: Scalar - Maximum expert usage percentage
    """
    # Handle tuple of router logits from multiple layers
    if isinstance(router_logits, tuple):
        router_logits = aggregate_router_logits_across_layers(router_logits, aggregation="mean")

    batch_size, seq_len, num_experts = router_logits.shape

    # Get top-k experts per token
    _, top_k_indices = torch.topk(router_logits, k=top_k, dim=-1)  # [batch, seq_len, k]

    # Count assignments per expert
    expert_counts = torch.zeros(num_experts, device=router_logits.device)

    # Flatten and count (only valid tokens)
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, top_k)  # [batch, seq_len, k]
    valid_indices = top_k_indices[attention_mask_expanded.bool()]  # [num_valid_tokens * k]

    for expert_id in range(num_experts):
        expert_counts[expert_id] = (valid_indices == expert_id).sum()

    # Compute statistics
    total_assignments = expert_counts.sum()
    expert_percentages = (expert_counts / total_assignments.clamp(min=1)) * 100

    stats = {
        "expert_counts": expert_counts.cpu().tolist(),
        "expert_percentages": expert_percentages.cpu().tolist(),
        "utilization_variance": expert_percentages.var().item(),
        "min_utilization": expert_percentages.min().item(),
        "max_utilization": expert_percentages.max().item(),
    }

    return stats
