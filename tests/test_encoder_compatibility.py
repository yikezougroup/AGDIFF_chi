"""Regression tests for the released CREMP checkpoint's shared edge encoder."""
import copy
from pathlib import Path
import unittest

import torch
from easydict import EasyDict
from src.agdiff.models.epsnet import get_model


class EncoderCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / 'logs/cremp_default_batch64_2024_12_12__14_42_15/best_model/best_model.pt'
        cls.ckpt = torch.load(path, map_location='cpu', weights_only=False)

    def make_model(self, mode=None):
        config = copy.deepcopy(self.ckpt['config'].model)
        if mode is not None:
            config.local_edge_encoder = mode
        model = get_model(config).eval()
        model.load_state_dict(self.ckpt['model'], strict=True)
        return model

    def forward(self, model):
        # Explicit edges isolate encoder routing from graph construction.
        return model(
            atom_type=torch.tensor([6, 7, 8]),
            pos=torch.tensor([[0., 0., 0.], [1.4, 0., 0.], [0., 1.2, 0.]]),
            bond_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
            bond_type=torch.ones(4, dtype=torch.long),
            batch=torch.zeros(3, dtype=torch.long), time_step=torch.tensor([0]),
            edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
            edge_type=torch.ones(4, dtype=torch.long),
            edge_length=torch.tensor([[1.4], [1.4], [1.2], [1.2]]),
        )

    def test_legacy_checkpoint_uses_shared_global_edge_encoder(self):
        model = self.make_model()
        with torch.no_grad():
            before = self.forward(model)
            for p in model.edge_encoder_local.parameters():
                p.add_(0.5)
            after = self.forward(model)
        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0,
            msg='Released CREMP local predictions must not use the untrained local edge encoder')

    def test_independent_encoder_is_explicit_opt_in(self):
        model = self.make_model('local')
        with torch.no_grad():
            before = self.forward(model)
            for p in model.edge_encoder_local.parameters():
                p.add_(0.5)
            after = self.forward(model)
        self.assertFalse(torch.equal(before[1], after[1]))
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)

    def test_invalid_encoder_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'local_edge_encoder'):
            self.make_model('typo')

    def test_strict_checkpoint_state_dict_compatibility(self):
        model = self.make_model()
        self.assertEqual(set(model.state_dict()), set(self.ckpt['model']))

    def sample(self, save_traj):
        model = self.make_model()
        torch.manual_seed(17)
        return model.langevin_dynamics_sample_diffusion(
            atom_type=torch.tensor([6, 7]), pos_init=torch.randn(2, 3),
            bond_index=torch.tensor([[0, 1], [1, 0]]),
            bond_type=torch.ones(2, dtype=torch.long), batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1, extend_order=False, n_steps=2, save_traj=save_traj)

    def test_disabling_trajectory_preserves_sample_and_stores_no_frames(self):
        with torch.no_grad():
            expected, frames = self.sample(True)
            actual, empty = self.sample(False)
        self.assertEqual(len(frames), 2)
        self.assertEqual(empty, [])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
