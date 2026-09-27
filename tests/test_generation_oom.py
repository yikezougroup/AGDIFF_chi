"""Entry-point OOM regressions with real CPU sampling and RDKit I/O/filtering."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.data import Data

from scripts import generate_filtered as generation
from src.agdiff.models.epsnet import get_model
from src.agdiff.models.epsnet.dualenc import DualEncoderEpsNetwork


class GenerationOOMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
        scratch.mkdir(parents=True, exist_ok=True)
        cls.root = tempfile.TemporaryDirectory(prefix='agdiff-oom-', dir=scratch)
        cls.addClassCleanup(cls.root.cleanup)
        cls.checkpoint = Path(cls.root.name) / 'tiny-model.pt'
        # A small, real network and schedule keep this an entry-point integration
        # test, not a substitute sampler or fabricated accepted coordinates.
        config = EasyDict(model=dict(
            type='diffusion', network='dualenc', hidden_dim=128, num_convs=1,
            num_convs_local=1, cutoff=10., mlp_act='relu', beta_schedule='linear',
            beta_start=.1, beta_end=.2, num_diffusion_timesteps=2,
            edge_order=3, edge_encoder='mlp', smooth_conv=True,
        ))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = get_model(config.model)
        torch.save({'config': config, 'model': model.state_dict()}, cls.checkpoint)

    def setUp(self):
        self.output_root = tempfile.TemporaryDirectory(dir=self.root.name)
        self.addCleanup(self.output_root.cleanup)
        self.out = Path(self.output_root.name) / 'output'

    def argv(self, target, batch_size):
        return [str(self.checkpoint), '--smiles', 'N#N', '--out', str(self.out),
                '--target', str(target),
                '--batch-size', str(batch_size), '--candidate-offset', '37',
                '--seed', '17', '--device', 'cpu', '--threads', '1',
                '--n-steps', '2', '--keep-raw']

    @contextlib.contextmanager
    def inject_oom(self, site, fail):
        """Inject only the failure; successful paths call the real implementation."""
        original_repeat = generation.repeat_data
        original_to = Data.to
        original_sample = DualEncoderEpsNetwork.langevin_dynamics_sample_diffusion
        state = {'attempts': [], 'sampled': [], 'failed_batches': []}

        def maybe_fail(size, batch=None):
            state['attempts'].append(size)
            if fail(size, len(state['attempts'])):
                if batch is not None:
                    state['failed_batches'].append(weakref.ref(batch))
                raise torch.cuda.OutOfMemoryError(f'injected {site} OOM')

        def repeat(data, size):
            # The preceding failed batch must be released before the next attempt.
            self.assertTrue(all(ref() is None for ref in state['failed_batches']))
            if site == 'construction':
                maybe_fail(size)
            return original_repeat(data, size)

        def transfer(batch, device, *args, **kwargs):
            if site == 'transfer':
                # PyG transfers attributes in place. Touch one tensor first to
                # reproduce a failure with a partially transferred live batch.
                batch.atom_type = batch.atom_type.to(device)
                maybe_fail(batch.num_graphs, batch)
            return original_to(batch, device, *args, **kwargs)

        def sample(model, *args, **kwargs):
            size = kwargs['num_graphs']
            if site == 'sampling':
                maybe_fail(size)
            result = original_sample(model, *args, **kwargs)
            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertEqual(result[1], [])
            state['sampled'].append(size)
            return result

        with mock.patch.object(generation, 'repeat_data', new=repeat), \
                mock.patch.object(Data, 'to', new=transfer), \
                mock.patch.object(DualEncoderEpsNetwork,
                                  'langevin_dynamics_sample_diffusion', new=sample), \
                mock.patch.object(generation.gc, 'collect', wraps=generation.gc.collect) as collect, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            state['collect'] = collect
            yield state

    def read_sdf(self, name):
        path = self.out / name
        molecules = list(Chem.SDMolSupplier(str(path), removeHs=False)) if path.stat().st_size else []
        self.assertNotIn(None, molecules)
        return molecules

    def assert_recovery(self, site, after_success=False):
        budget = 12 if after_success else 7
        expected_sizes = [5, 2, 1, 2, 2] if after_success else [2, 1, 2, 2]
        expected_attempts = [5, 5, 2, 3, 1, 2, 2] if after_success else [5, 2, 3, 1, 2, 2]
        with self.inject_oom(site, lambda size, call: size > 2 and (not after_success or call > 1)) as state:
            status = generation.main(self.argv(10 if after_success else 6, 5))
        self.assertEqual(state['attempts'], expected_attempts)
        self.assertEqual(state['sampled'], expected_sizes)
        self.assertEqual(state['collect'].call_count, 2)
        self.assertTrue(all(ref() is None for ref in state['failed_batches']))
        summary = json.loads((self.out / 'summary.json').read_text())
        progress = json.loads((self.out / 'progress.json').read_text())
        target = 10 if after_success else 6
        success = summary['written'] == target
        self.assertEqual(status, 0 if success else 2)
        self.assertEqual(summary['success'], success)
        self.assertEqual(summary['chiral_center_count'], 0)
        self.assertEqual(summary['stop_reason'], 'target_met' if success else 'candidate_budget_exhausted')
        self.assertEqual(summary['candidate_budget'], budget)
        self.assertEqual(summary['generated'], budget)
        self.assertEqual(summary['oom_retries'], 2)
        self.assertEqual([b['size'] for b in summary['batch_sizes']], expected_sizes)
        for key in ('generated', 'accepted', 'written', 'oom_retries'):
            self.assertEqual(progress[key], summary[key])
        self.assertEqual(progress['batches'], summary['batch_sizes'])
        raw = self.read_sdf('raw.sdf')
        accepted = self.read_sdf('molecules.sdf')
        self.assertEqual([m.GetIntProp('agdiff_candidate_id') for m in raw],
                         list(range(37, 37 + budget)))
        # N#N has no stereocenters: independently audit the real bond-length gate
        # to check accepted IDs against raw IDs, not merely against their count.
        expected_ids = []
        for mol in raw:
            self.assertEqual(mol.GetIntProp('agdiff_seed'), 17)
            xyz = mol.GetConformer().GetPositions()
            self.assertTrue(np.isfinite(xyz).all())
            if .8 < np.linalg.norm(xyz[0] - xyz[1]) < 2.:
                expected_ids.append(mol.GetIntProp('agdiff_candidate_id'))
        self.assertTrue(expected_ids, 'Fixture must exercise accepted output IDs')
        self.assertEqual([m.GetIntProp('agdiff_candidate_id') for m in accepted], expected_ids[:target])
        self.assertEqual(summary['accepted'], len(expected_ids))
        self.assertEqual(summary['written'], len(accepted))
        self.assertEqual(summary['rejected'], budget - len(expected_ids))
        for mol in accepted:
            self.assertEqual(mol.GetIntProp('agdiff_seed'), 17)
        self.assertFalse((self.out / '.chunk_raw.sdf').exists())
        self.assertFalse((self.out / '.chunk_filtered.sdf').exists())

    def test_construction_oom_preserves_budget_and_ids(self):
        self.assert_recovery('construction')

    def test_transfer_oom_preserves_budget_and_ids(self):
        self.assert_recovery('transfer')

    def test_sampling_oom_preserves_budget_and_ids(self):
        self.assert_recovery('sampling')

    def test_construction_oom_after_success_preserves_budget_and_ids(self):
        self.assert_recovery('construction', after_success=True)

    def test_transfer_oom_after_success_preserves_budget_and_ids(self):
        self.assert_recovery('transfer', after_success=True)

    def test_sampling_oom_after_success_preserves_budget_and_ids(self):
        self.assert_recovery('sampling', after_success=True)

    def assert_singleton_failure(self, site):
        with self.inject_oom(site, lambda size, call: True) as state:
            with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, f'injected {site} OOM'):
                generation.main(self.argv(3, 3))
        self.assertEqual(state['attempts'], [3, 1], 'A singleton must fail, not split or loop')
        self.assertEqual(state['sampled'], [])
        self.assertEqual(state['collect'].call_count, 2)
        self.assertTrue(all(ref() is None for ref in state['failed_batches']))
        self.assertFalse((self.out / 'summary.json').exists())
        self.assertFalse((self.out / 'progress.json').exists())
        self.assertEqual(self.read_sdf('raw.sdf'), [])
        self.assertEqual(self.read_sdf('molecules.sdf'), [])

    def test_singleton_construction_oom_is_reraised(self):
        self.assert_singleton_failure('construction')

    def test_singleton_transfer_oom_is_reraised(self):
        self.assert_singleton_failure('transfer')

    def test_singleton_sampling_oom_is_reraised(self):
        self.assert_singleton_failure('sampling')


if __name__ == '__main__':
    unittest.main()
