"""
test_supervised_routing.py

Unit test for supervised routing module with dummy MoE model.
Tests all components without requiring full Mixtral model download.

Run:
    python test_supervised_routing.py
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
import sys


def create_dummy_moe_model(batch_size=2, seq_len=10, num_experts=8, num_layers=4):
    """Create a dummy MoE model that mimics Mixtral's output structure."""

    class DummyMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 32000
            self.hidden_size = 4096
            self.num_experts = num_experts
            self.num_layers = num_layers

        def forward(self, input_ids, attention_mask, labels, output_router_logits=True, **kwargs):
            batch_size, seq_len = input_ids.shape

            # Dummy logits
            logits = torch.randn(batch_size, seq_len, self.vocab_size)

            # Compute dummy loss (cross-entropy)
            loss = torch.tensor(2.5, requires_grad=True)

            # Dummy router logits (tuple of tensors, one per MoE layer)
            router_logits = None
            if output_router_logits:
                router_logits = tuple([
                    torch.randn(batch_size, seq_len, self.num_experts, requires_grad=True)
                    for _ in range(self.num_layers)
                ])

            return MoeCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                router_logits=router_logits,
            )

    return DummyMoEModel()


def test_expert_label_embedding():
    """Test ExpertLabelEmbedding module."""
    print("\n[Test 1] ExpertLabelEmbedding...")

    from supervised_routing import ExpertLabelEmbedding

    # Create embedding layer
    embedding = ExpertLabelEmbedding(
        num_categories=4,
        num_experts=8,
        temperature=1.0,
    )

    # Test forward pass
    label_ids = torch.tensor([0, 1, 2, 3, 0, 1])  # batch_size=6
    expert_prefs = embedding(label_ids)

    # Verify shape
    assert expert_prefs.shape == (6, 8), f"Expected shape (6, 8), got {expert_prefs.shape}"

    # Verify output is reasonable (not all zeros, not NaN)
    assert not torch.isnan(expert_prefs).any(), "Expert preferences contain NaN"
    assert expert_prefs.abs().sum() > 0, "Expert preferences are all zero"

    print(f"  ✓ Shape: {expert_prefs.shape}")
    print(f"  ✓ No NaN values")
    print(f"  ✓ Sample preferences: {expert_prefs[0][:4].tolist()}")

    return True


def test_routing_supervision_loss():
    """Test routing supervision loss computation."""
    print("\n[Test 2] Routing Supervision Loss...")

    from supervised_routing import compute_routing_supervision_loss

    batch_size, seq_len, num_experts = 2, 10, 8

    # Create dummy inputs
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    expert_prefs = torch.randn(batch_size, num_experts)
    attention_mask = torch.ones(batch_size, seq_len)

    # Compute loss
    loss = compute_routing_supervision_loss(
        router_logits=router_logits,
        expert_prefs=expert_prefs,
        attention_mask=attention_mask,
    )

    # Verify loss properties
    assert loss.numel() == 1, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"  ✓ Loss is non-negative")
    print(f"  ✓ Loss is scalar")

    # Test with partial attention mask
    attention_mask[0, 5:] = 0  # Mask out second half of first sequence
    loss_masked = compute_routing_supervision_loss(
        router_logits=router_logits,
        expert_prefs=expert_prefs,
        attention_mask=attention_mask,
    )

    print(f"  ✓ Masked loss computed: {loss_masked.item():.4f}")

    return True


def test_router_logits_aggregation():
    """Test router logits aggregation across layers."""
    print("\n[Test 3] Router Logits Aggregation...")

    from supervised_routing import aggregate_router_logits_across_layers

    batch_size, seq_len, num_experts = 2, 10, 8
    num_layers = 4

    # Create tuple of router logits
    router_logits_tuple = tuple([
        torch.randn(batch_size, seq_len, num_experts)
        for _ in range(num_layers)
    ])

    # Test mean aggregation
    aggregated_mean = aggregate_router_logits_across_layers(
        router_logits_tuple,
        aggregation="mean"
    )
    assert aggregated_mean.shape == (batch_size, seq_len, num_experts)
    print(f"  ✓ Mean aggregation shape: {aggregated_mean.shape}")

    # Test last aggregation
    aggregated_last = aggregate_router_logits_across_layers(
        router_logits_tuple,
        aggregation="last"
    )
    assert aggregated_last.shape == (batch_size, seq_len, num_experts)
    assert torch.equal(aggregated_last, router_logits_tuple[-1])
    print(f"  ✓ Last aggregation correct")

    # Test sum aggregation
    aggregated_sum = aggregate_router_logits_across_layers(
        router_logits_tuple,
        aggregation="sum"
    )
    assert aggregated_sum.shape == (batch_size, seq_len, num_experts)
    print(f"  ✓ Sum aggregation shape: {aggregated_sum.shape}")

    return True


def test_supervised_moe_wrapper():
    """Test SupervisedMoEWrapper with dummy model."""
    print("\n[Test 4] SupervisedMoEWrapper...")

    from supervised_routing import SupervisedMoEWrapper

    # Create dummy model
    dummy_model = create_dummy_moe_model()

    # Wrap with supervised routing
    wrapped_model = SupervisedMoEWrapper(
        moe_model=dummy_model,
        num_categories=4,
        num_experts=8,
        routing_loss_weight=0.1,
    )

    print(f"  ✓ Wrapper initialized")

    # Create dummy batch
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 32000, (batch_size, seq_len))
    expert_label_ids = torch.randint(0, 4, (batch_size,))

    # Forward pass WITH expert labels
    outputs = wrapped_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        expert_label_ids=expert_label_ids,
    )

    # Verify outputs
    assert outputs.loss is not None, "Loss is None"
    assert hasattr(outputs, 'routing_supervision_loss'), "Routing loss not in outputs"
    assert outputs.routing_supervision_loss is not None, "Routing supervision loss is None"

    lm_loss = outputs.loss.item()
    routing_loss = outputs.routing_supervision_loss.item()

    print(f"  ✓ Combined loss: {lm_loss:.4f}")
    print(f"  ✓ Routing supervision loss: {routing_loss:.4f}")
    print(f"  ✓ Loss includes routing supervision")

    # Forward pass WITHOUT expert labels (should work without routing loss)
    outputs_no_labels = wrapped_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        expert_label_ids=None,
    )

    assert outputs_no_labels.loss is not None, "Loss is None without expert labels"
    print(f"  ✓ Works without expert labels")

    # Test backward pass
    outputs.loss.backward()
    print(f"  ✓ Backward pass successful")

    return True


def test_expert_utilization_stats():
    """Test expert utilization statistics computation."""
    print("\n[Test 5] Expert Utilization Statistics...")

    from supervised_routing import get_expert_utilization_stats

    batch_size, seq_len, num_experts = 4, 20, 8
    top_k = 2

    # Create router logits with some imbalance
    router_logits = torch.randn(batch_size, seq_len, num_experts)
    # Make expert 0 and 1 more likely to be selected
    router_logits[:, :, 0] += 1.0
    router_logits[:, :, 1] += 0.5

    attention_mask = torch.ones(batch_size, seq_len)

    # Compute stats
    stats = get_expert_utilization_stats(
        router_logits=router_logits,
        attention_mask=attention_mask,
        top_k=top_k,
    )

    # Verify stats structure
    assert 'expert_counts' in stats, "Missing expert_counts"
    assert 'expert_percentages' in stats, "Missing expert_percentages"
    assert 'utilization_variance' in stats, "Missing utilization_variance"
    assert 'min_utilization' in stats, "Missing min_utilization"
    assert 'max_utilization' in stats, "Missing max_utilization"

    print(f"  ✓ Expert counts: {[int(c) for c in stats['expert_counts']]}")
    print(f"  ✓ Expert percentages: {[f'{p:.1f}%' for p in stats['expert_percentages']]}")
    print(f"  ✓ Utilization variance: {stats['utilization_variance']:.2f}")
    print(f"  ✓ Min utilization: {stats['min_utilization']:.1f}%")
    print(f"  ✓ Max utilization: {stats['max_utilization']:.1f}%")

    # Verify percentages sum to ~100% * top_k (since each token routes to top_k experts)
    total_percentage = sum(stats['expert_percentages'])
    expected_total = 100.0 * top_k  # Each token contributes to top_k experts
    assert abs(total_percentage - expected_total) < 1.0, \
        f"Percentages should sum to ~{expected_total}, got {total_percentage}"

    print(f"  ✓ Percentages sum correctly: {total_percentage:.1f}% (expected ~{expected_total}%)")

    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("Supervised Routing Module Unit Tests")
    print("=" * 70)

    tests = [
        ("Expert Label Embedding", test_expert_label_embedding),
        ("Routing Supervision Loss", test_routing_supervision_loss),
        ("Router Logits Aggregation", test_router_logits_aggregation),
        ("Supervised MoE Wrapper", test_supervised_moe_wrapper),
        ("Expert Utilization Stats", test_expert_utilization_stats),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("=" * 70)

    if failed == 0:
        print("\nALL TESTS PASSED! ✓")
        print("\nNext steps:")
        print("1. Run model loading test: python test_model_loading.py")
        print("2. Run integration test with real data")
        print("3. Start training: accelerate launch train_mixtral_8x7b_moe_accelerate.py")
        return True
    else:
        print(f"\n{failed} TEST(S) FAILED ✗")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
