"""Target-loop control tests: synthetic sampler, real RDKit writer and QC.

These fixtures test accounting/stopping, not learned-model inference quality.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts import generate_filtered as generation


class TargetGenerationTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
        self.addCleanup(self.root.cleanup)
        self.out = Path(self.root.name) / 'output'
        self.ckpt = Path(self.root.name) / 'fixture.pt'
        self.ckpt.write_bytes(b'control-flow fixture, not model weights')

    def argv(self, target=5, size=4, smiles='N#N'):
        return [str(self.ckpt), '--smiles', smiles, '--out', str(self.out),
                '--target', str(target), '--batch-size', str(size), '--seed', '17',
                '--device', 'cpu', '--n-steps', '1', '--keep-raw',
                '--candidate-offset', '2147483647']

    def run_generation(self, target=5, size=4, smiles='N#N', passing=True, oom=False):
        attempts = []
        fixture = None
        if passing and smiles != 'N#N':
            fixture = Chem.AddHs(Chem.MolFromSmiles(smiles))
            self.assertEqual(AllChem.EmbedMolecule(fixture, randomSeed=17), 0)
            fixture = torch.tensor(fixture.GetConformer().GetPositions(), dtype=torch.float32)
        def sample(**kwargs):
            count = kwargs['num_graphs']
            attempts.append(count)
            if oom and len(attempts) == 1:
                raise torch.cuda.OutOfMemoryError('injected control-flow OOM')
            positions = torch.zeros_like(kwargs['pos_init'])
            if fixture is not None:
                positions = fixture.repeat(count, 1)
            elif passing:
                positions[1::2, 0] = 1.1  # Valid N#N fixture only.
            return positions, []
        model = mock.Mock(local_edge_encoder='global')
        model.to.return_value = model
        model.eval.return_value = model
        model.langevin_dynamics_sample_diffusion.side_effect = sample
        checkpoint = {'config': SimpleNamespace(model=SimpleNamespace(
            num_diffusion_timesteps=1, edge_order=3)), 'model': {}}
        with mock.patch.object(generation.torch, 'load', return_value=checkpoint), \
                mock.patch.object(generation, 'get_model', return_value=model), \
                contextlib.redirect_stdout(io.StringIO()):
            status = generation.main(self.argv(target, size, smiles))
        summary = json.loads((self.out / 'summary.json').read_text())
        progress = json.loads((self.out / 'progress.json').read_text())
        for key in ('chiral_center_count', 'candidate_budget', 'stop_reason', 'generated', 'written'):
            self.assertEqual(progress[key], summary[key])
        self.assertFalse(summary['force_field_optimization'])
        self.assertFalse(summary['save_traj'])
        return status, summary, attempts

    def test_success_at_cap_writes_only_target_and_counts_excess_passes(self):
        status, summary, attempts = self.run_generation(target=10, size=4)
        self.assertEqual(status, 0)
        self.assertEqual(attempts, [4, 4, 4])
        self.assertEqual(summary['written'], 10)
        self.assertEqual(summary['accepted'], 12)
        self.assertEqual(summary['stop_reason'], 'target_met')

    def test_early_success_before_cap(self):
        status, summary, attempts = self.run_generation(target=20, size=5)
        self.assertEqual(status, 0)
        self.assertEqual(attempts, [5, 5, 5, 5])
        self.assertEqual(summary['candidate_budget'], 24)
        self.assertEqual(summary['generated'], 20)

    def test_rejections_exhaust_cap_with_exact_tail(self):
        status, summary, attempts = self.run_generation(target=6, size=5, passing=False)
        self.assertEqual(status, 2)
        self.assertEqual(attempts, [5, 2])
        self.assertEqual(summary['candidate_budget'], 7)
        self.assertEqual(summary['generated'], 7)
        self.assertEqual(summary['written'], 0)
        self.assertEqual(summary['chiral_center_count'], 0)
        self.assertEqual(summary['stop_reason'], 'candidate_budget_exhausted')

    def test_chiral_cap_counts_assigned_and_unassigned_centers(self):
        for smiles, count, budget in [('FC(Cl)Br', 1, 2), ('F[C@H](Cl)Br', 1, 2),
                                       ('F[C@@H](Cl)Br', 1, 2),
                                       ('CC(O)C(F)Cl', 2, 4)]:
            with self.subTest(smiles=smiles):
                self.out = Path(self.root.name) / str(budget) / smiles
                status, summary, attempts = self.run_generation(
                    target=1, size=3, smiles=smiles, passing=False)
                self.assertEqual(status, 2)
                self.assertEqual(summary['chiral_center_count'], count)
                self.assertEqual(summary['candidate_budget'], budget)
                self.assertEqual(sum(attempts), budget)

    def test_huge_cap_is_lazy_before_first_sampling_attempt(self):
        # A real connected graph with many unassigned centers. Stop the probe
        # at batch construction rather than attempting huge molecular sampling.
        smiles = 'FC(Cl)' + 'C(F)(Cl)' * 35 + 'Br'
        # Scheduling laziness is guarded directly: deque may hold failed splits,
        # never the full batch iterator (which would exhaust memory here).
        original_deque = generation.deque
        def bounded_deque(iterable=()):
            iterator = iter(iterable)
            values = []
            for value in iterator:
                values.append(value)
                self.assertLess(len(values), 10, 'Eager allocation of exponential batch schedule')
            return original_deque(values)
        with mock.patch.object(generation, 'deque', side_effect=bounded_deque), \
                mock.patch.object(generation, 'repeat_data', side_effect=RuntimeError('first lazy batch')):
            with self.assertRaisesRegex(RuntimeError, 'first lazy batch'):
                self.run_generation(target=1, size=1, smiles=smiles, passing=False)

    def test_oom_splits_preserve_cap_and_success(self):
        status, summary, attempts = self.run_generation(target=5, size=6, oom=True)
        self.assertEqual(status, 0)
        self.assertEqual(attempts, [6, 3, 3])
        self.assertEqual(summary['oom_retries'], 1)
        self.assertEqual(summary['generated'], 6)
        self.assertEqual(summary['written'], 5)

    def test_oom_pending_split_is_abandoned_after_target(self):
        status, summary, attempts = self.run_generation(
            target=5, size=12, smiles='F[C@H](Cl)Br', oom=True)
        self.assertEqual(status, 0)
        self.assertEqual(attempts, [12, 6])
        self.assertEqual(summary['candidate_budget'], 12)
        self.assertEqual(summary['generated'], 6)
        self.assertEqual(summary['accepted'], 6)
        self.assertEqual(summary['written'], 5)
        self.assertEqual(summary['stop_reason'], 'target_met')

    def test_one_target_caps_initial_batch_to_one_raw_candidate(self):
        status, summary, attempts = self.run_generation(target=1, size=128)
        self.assertEqual(status, 0)
        self.assertEqual(attempts, [1])
        self.assertEqual(summary['candidate_budget'], 1)

    def test_invalid_arguments_fail_before_checkpoint_or_output(self):
        cases = [(['--target', '0'], 'target'), (['--target', '-1'], 'target'),
                 (['--batch-size', '0'], 'batch'), (['--batch-size', '-2'], 'batch'),
                 (['--candidate-offset', '-1'], 'offset'),
                 (['--threads', '0'], 'threads'),
                 (['--target', '1.5'], 'invalid int'),
                 (['--smiles', 'N.N'], 'connected molecule'),
                 (['--smiles', ''], 'connected molecule'),
                 (['--candidates', '10'], 'unrecognized arguments'),
                 (['--early-stop'], 'unrecognized arguments')]
        for extra, message in cases:
            with self.subTest(extra=extra), mock.patch.object(generation.torch, 'load') as load:
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    generation.main(self.argv() + extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                load.assert_not_called()
                self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
