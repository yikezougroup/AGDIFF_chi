"""Metadata boundary regressions with real CPU sampling and RDKit entry-point I/O."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from easydict import EasyDict
from rdkit import Chem

from scripts import generate_filtered as generation
from src.agdiff.models.epsnet import get_model


class GenerationMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
        scratch.mkdir(parents=True, exist_ok=True)
        cls.root = tempfile.TemporaryDirectory(prefix='agdiff-metadata-', dir=scratch)
        cls.addClassCleanup(cls.root.cleanup)
        cls.checkpoint = Path(cls.root.name) / 'tiny-model.pt'
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

    def read_sdf(self, path):
        molecules = list(Chem.SDMolSupplier(str(path), removeHs=False)) if path.stat().st_size else []
        self.assertNotIn(None, molecules)
        return molecules

    def assert_metadata(self, seed, offset, keep_raw):
        # A real tiny model samples multiple batches; neither the sampler nor
        # the RDKit writer/filter is mocked, and no coordinates are fabricated.
        budget = 12
        with tempfile.TemporaryDirectory(dir=self.root.name) as directory:
            out = Path(directory) / 'output'
            argv = [str(self.checkpoint), '--smiles', 'N#N', '--out', str(out),
                    '--target', '0', '--candidates', str(budget), '--batch-size', '5',
                    '--candidate-offset', str(offset), '--seed', str(seed),
                    '--device', 'cpu', '--threads', '1', '--n-steps', '2']
            if keep_raw:
                argv.append('--keep-raw')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(generation.main(argv), 0)

            accepted = self.read_sdf(out / 'molecules.sdf')
            self.assertTrue(accepted, 'Fixture must exercise accepted metadata writes')
            accepted_ids = [int(m.GetProp('agdiff_candidate_id')) for m in accepted]
            expected_ids = list(range(offset, offset + budget))
            self.assertEqual(accepted_ids, sorted(set(accepted_ids)))
            self.assertTrue(set(accepted_ids).issubset(expected_ids))
            if offset == 2147483647:
                self.assertTrue(any(value > 2147483647 for value in accepted_ids),
                                'Fixture must accept an ID above signed32')

            all_molecules = list(accepted)
            if keep_raw:
                raw = self.read_sdf(out / 'raw.sdf')
                self.assertEqual(len(raw), budget)
                self.assertEqual([m.GetProp('agdiff_candidate_id') for m in raw],
                                 [str(value) for value in expected_ids])
                # N#N is achiral, so audit the real bond gate independently to
                # verify filtered IDs still refer to their original candidates.
                passing_ids = [int(m.GetProp('agdiff_candidate_id')) for m in raw
                               if .8 < np.linalg.norm(np.diff(m.GetConformer().GetPositions(), axis=0)) < 2.]
                self.assertEqual(accepted_ids, passing_ids)
                all_molecules.extend(raw)
            else:
                self.assertFalse((out / 'raw.sdf').exists())

            for mol in all_molecules:
                candidate_id = int(mol.GetProp('agdiff_candidate_id'))
                self.assertEqual(mol.GetProp('agdiff_candidate_id'), str(candidate_id))
                self.assertEqual(mol.GetProp('agdiff_seed'), str(seed))
                self.assertTrue(np.isfinite(mol.GetConformer().GetPositions()).all())
                # Decimal-string storage must preserve existing integer reads
                # for metadata that fits the old signed32 representation.
                if candidate_id <= 2147483647:
                    self.assertEqual(mol.GetIntProp('agdiff_candidate_id'), candidate_id)
                if seed <= 2147483647:
                    self.assertEqual(mol.GetIntProp('agdiff_seed'), seed)

            summary = json.loads((out / 'summary.json').read_text())
            progress = json.loads((out / 'progress.json').read_text())
            self.assertTrue(summary['success'])
            self.assertEqual(summary['target'], 0)
            self.assertEqual(summary['seed'], seed)
            self.assertEqual(summary['candidate_offset'], offset)
            self.assertEqual(summary['candidate_budget'], budget)
            self.assertEqual(summary['generated'], budget)
            self.assertEqual(summary['accepted'], len(accepted))
            self.assertEqual(summary['written'], len(accepted))
            self.assertEqual(summary['rejected'], budget - len(accepted))
            self.assertEqual(summary['oom_retries'], 0)
            self.assertEqual(summary['filter_reasons'], {'specified_centers_match': len(accepted)})
            expected_batches = []
            start = offset
            for size in (5, 5, 2):
                count = sum(start <= value < start + size for value in accepted_ids)
                expected_batches.append({'size': size, 'accepted': count, 'written': count})
                start += size
            self.assertEqual(summary['batch_sizes'], expected_batches)
            for key in ('generated', 'accepted', 'written', 'oom_retries'):
                self.assertEqual(progress[key], summary[key])
            self.assertEqual(progress['batches'], expected_batches)
            results = [json.loads(line.removeprefix('RESULT '))
                       for line in stdout.getvalue().splitlines() if line.startswith('RESULT ')]
            self.assertEqual(results, [summary])
            self.assertFalse((out / '.chunk_raw.sdf').exists())
            self.assertFalse((out / '.chunk_filtered.sdf').exists())

    def test_large_seed_with_raw_output(self):
        for seed in (2147483648, 4294967295):
            with self.subTest(seed=seed):
                self.assert_metadata(seed, offset=37, keep_raw=True)

    def test_large_seed_without_raw_output(self):
        for seed in (2147483648, 4294967295):
            with self.subTest(seed=seed):
                self.assert_metadata(seed, offset=37, keep_raw=False)

    def test_candidate_ids_cross_signed32_with_raw_output(self):
        self.assert_metadata(seed=17, offset=2147483647, keep_raw=True)

    def test_candidate_ids_cross_signed32_without_raw_output(self):
        self.assert_metadata(seed=17, offset=2147483647, keep_raw=False)

    def test_in_range_metadata_with_raw_output_preserves_integer_reads(self):
        for seed in (17, 2147483647):
            with self.subTest(seed=seed):
                self.assert_metadata(seed, offset=37, keep_raw=True)

    def test_in_range_metadata_without_raw_output_preserves_integer_reads(self):
        for seed in (17, 2147483647):
            with self.subTest(seed=seed):
                self.assert_metadata(seed, offset=37, keep_raw=False)


if __name__ == '__main__':
    unittest.main()
