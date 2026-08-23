import unittest

try:
    import torch

    from train_nnue.expert_blending_model import NNUEExperts, transform_gate_logits
    from train_nnue.train_expert_blending import (
        compute_gate_statistics,
        compute_router_statistics,
        router_teacher_distribution,
    )
except ModuleNotFoundError:
    torch = None
    NNUEExperts = None
    transform_gate_logits = None
    compute_gate_statistics = None
    compute_router_statistics = None
    router_teacher_distribution = None


@unittest.skipIf(torch is None, "requires the nnue PyTorch environment")
class ForwardExpertTest(unittest.TestCase):
    def test_softmax_transform_is_backward_compatible(self):
        logits = torch.randn(3, 8)
        torch.testing.assert_close(
            transform_gate_logits(logits, "softmax"),
            torch.nn.functional.softmax(logits, dim=-1),
        )

    def test_entmax15_is_sparse_and_normalized(self):
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, -1.0, -2.0]],
            requires_grad=True,
        )
        weights = transform_gate_logits(logits, "entmax15")
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(weights[0], torch.full((4,), 0.25))
        self.assertGreaterEqual(int((weights[1] == 0).sum()), 3)
        weights.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_matches_one_hot_blending(self):
        torch.manual_seed(7)
        batch_size = 3
        num_features = 4
        us = torch.cat(
            [torch.ones(batch_size, 256), torch.zeros(batch_size, 256)], dim=1
        )
        them = 1.0 - us
        white = torch.randn(batch_size, num_features)
        black = torch.randn(batch_size, num_features)

        for mode in ("weighted", "residual"):
            model = NNUEExperts(2, num_features, blend_mode=mode)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.03)
            for expert in range(2):
                gate = torch.nn.functional.one_hot(
                    torch.full((batch_size,), expert), num_classes=2
                ).float()
                blended = model(gate, us, them, white, black)
                direct = model.forward_expert(
                    expert, us, them, white, black
                )
                torch.testing.assert_close(
                    blended, direct, rtol=1e-5, atol=1e-6
                )

    def test_gate_regularization_terms(self):
        uniform = torch.full((4, 8), 1 / 8)
        uniform_statistics = compute_gate_statistics(uniform)
        self.assertAlmostEqual(
            uniform_statistics["entropy"].item(),
            torch.log(torch.tensor(8.0)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            uniform_statistics["balance_kl"].item(), 0.0, places=6
        )
        self.assertAlmostEqual(
            uniform_statistics["effective_experts"].item(), 8.0, places=5
        )

        collapsed = torch.zeros((4, 8))
        collapsed[:, 0] = 1.0
        collapsed_statistics = compute_gate_statistics(collapsed)
        self.assertAlmostEqual(
            collapsed_statistics["entropy"].item(), 0.0, places=6
        )
        self.assertAlmostEqual(
            collapsed_statistics["balance_kl"].item(),
            torch.log(torch.tensor(8.0)).item(),
            places=6,
        )

    def test_router_teacher_and_statistics(self):
        losses = torch.tensor([[0.3, 0.1, 0.2], [0.0, 0.4, 0.2]])
        hard = router_teacher_distribution(losses, "hard")
        torch.testing.assert_close(
            hard, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )
        soft = router_teacher_distribution(losses, "soft", temperature=0.05)
        torch.testing.assert_close(soft.sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(soft.argmax(dim=-1), losses.argmin(dim=-1))

        gates = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2]])
        statistics = compute_router_statistics(gates, losses, hard)
        self.assertAlmostEqual(statistics["top1_match"].item(), 0.5)
        self.assertAlmostEqual(statistics["teacher_top1_match"].item(), 0.5)
        self.assertAlmostEqual(statistics["regret"].item(), 0.2)

        cached_teacher = torch.tensor(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]
        )
        cached_statistics = compute_router_statistics(
            gates, losses, cached_teacher
        )
        self.assertAlmostEqual(
            cached_statistics["teacher_top1_match"].item(), 0.5
        )


if __name__ == "__main__":
    unittest.main()
