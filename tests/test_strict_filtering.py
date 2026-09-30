"""Fail-closed geometry/CIP regressions using real RDKit conformers (CPU only)."""
import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts import smiles_generation as generation


ALL_S = 'N[C@@H](C)C(=O)N[C@@H](C)C(=O)O'
MIXED = 'N[C@H](C)C(=O)N[C@@H](C)C(=O)O'
PARTLY_SPECIFIED = 'N[C@@H](C)C(=O)NC(C)C(=O)O'
EZ_DISTINGUISHED = r'F[C@H](/C=C/C)/C=C\C'


def embed(smiles, seed=17):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise AssertionError('RDKit fixture embedding failed')
    return mol


def independently_perceived_centers(mol):
    """No production helpers or retained template stereo in the output audit."""
    probe = Chem.Mol(mol)
    Chem.RemoveStereochemistry(probe)
    probe.GetConformer().Set3D(True)
    Chem.AssignStereochemistryFrom3D(probe, replaceExistingTags=True)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    return dict(Chem.FindMolChiralCenters(probe, includeUnassigned=True))


class StrictFilteringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_s = embed(ALL_S)
        cls.specified = {1: 'S', 6: 'S'}
        cls.all_r = Chem.Mol(cls.all_s)
        for index, (x, y, z) in enumerate(cls.all_r.GetConformer().GetPositions()):
            cls.all_r.GetConformer().SetAtomPosition(index, (-x, y, z))
        cls.mixed = embed(MIXED)
        # Retain an all-S template on deliberately incompatible coordinates.
        for atom in cls.mixed.GetAtoms():
            template = cls.all_s.GetAtomWithIdx(atom.GetIdx())
            atom.SetChiralTag(template.GetChiralTag())
            if template.HasProp('_CIPCode'):
                atom.SetProp('_CIPCode', template.GetProp('_CIPCode'))
        cls.planar = Chem.Mol(cls.all_s)
        AllChem.Compute2DCoords(cls.planar)
        # RDKit depiction layouts differ across versions. Normalize the shortest
        # bond to 1 A so this fixture isolates planar chirality, not bond QC.
        conf = cls.planar.GetConformer()
        xyz = conf.GetPositions()
        shortest = min(np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()])
                       for b in cls.planar.GetBonds())
        for index, point in enumerate(xyz / shortest):
            conf.SetAtomPosition(index, tuple(point))

    def run_filter(self, molecules, smiles=ALL_S, num_needed=100):
        scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='agdiff-strict-', dir=scratch) as tmp:
            source, target = Path(tmp) / 'Final.sdf', Path(tmp) / 'Filter.sdf'
            with Chem.SDWriter(str(source)) as writer:
                for mol in molecules:
                    writer.write(mol)
            with contextlib.redirect_stdout(io.StringIO()):
                result = generation.filter_sdf_by_chirality(
                    str(source), str(target), smiles, num_needed)
            self.assertTrue(target.exists(), 'Even an empty filter must create output')
            saved = (list(Chem.SDMolSupplier(str(target), removeHs=False))
                     if target.stat().st_size else [])
            self.assertNotIn(None, saved)
            self.assertEqual(result[0], len(saved))
            return result, saved

    def test_all_s_is_accepted_and_independently_rechecked(self):
        self.assertEqual(independently_perceived_centers(self.all_s), self.specified)
        result, saved = self.run_filter([self.all_s])
        self.assertEqual(result, (1, [0]))
        self.assertEqual(independently_perceived_centers(saved[0]), self.specified)
        self.assertEqual(saved[0].GetProp('agdiff_filter_reason'), 'specified_centers_match')

    def test_all_r_is_reflected_and_independently_rechecked(self):
        self.assertEqual(independently_perceived_centers(self.all_r), {1: 'R', 6: 'R'})
        original = self.all_r.GetConformer().GetPositions().copy()
        result, saved = self.run_filter([self.all_r])
        self.assertEqual(result, (1, [0]))
        self.assertEqual(independently_perceived_centers(saved[0]), self.specified)
        self.assertEqual(saved[0].GetProp('agdiff_filter_reason'),
                         'all_specified_centers_reversed_flipped_to_match')
        np.testing.assert_array_equal(self.all_r.GetConformer().GetPositions(), original)

    def test_mixed_rs_is_rejected_despite_matching_template_tags(self):
        self.assertEqual(independently_perceived_centers(self.mixed), {1: 'R', 6: 'S'})
        self.assertFalse(generation._agdiff_chirality_matches(self.mixed, self.specified)[0])
        result, _ = self.run_filter([self.mixed])
        self.assertEqual(result, (0, []))

    def test_ez_distinguished_ligands_are_perceived_freshly(self):
        mol = embed(EZ_DISTINGUISHED, seed=5)
        expected = {1: 'R'}
        self.assertEqual(generation._agdiff_target_chiral_centers(EZ_DISTINGUISHED)[1], expected)
        # Neither atom nor double-bond template tags may be required to pass.
        Chem.RemoveStereochemistry(mol)
        self.assertEqual(independently_perceived_centers(mol), expected)
        result, saved = self.run_filter([mol], smiles=EZ_DISTINGUISHED)
        self.assertEqual(result, (1, [0]))
        self.assertEqual(independently_perceived_centers(saved[0]), expected)
        self.assertEqual(saved[0].GetProp('agdiff_filter_reason'), 'specified_centers_match')

    def test_ez_distinguished_mirror_is_reflected_and_rechecked(self):
        mol = embed(EZ_DISTINGUISHED, seed=5)
        for index, (x, y, z) in enumerate(mol.GetConformer().GetPositions()):
            mol.GetConformer().SetAtomPosition(index, (-x, y, z))
        self.assertEqual(independently_perceived_centers(mol), {1: 'S'})
        result, saved = self.run_filter([mol], smiles=EZ_DISTINGUISHED)
        self.assertEqual(result, (1, [0]))
        self.assertEqual(independently_perceived_centers(saved[0]), {1: 'R'})
        self.assertEqual(saved[0].GetProp('agdiff_filter_reason'),
                         'all_specified_centers_reversed_flipped_to_match')

    def test_identical_alkene_ligands_reject_stale_ez_template(self):
        mol = embed(EZ_DISTINGUISHED, seed=5)
        achiral = embed('FC(/C=C/C)/C=C/C', seed=5)
        # Keep the target's E/Z and atom tags but replace its coordinates with
        # two E ligands. Fresh perception must find no target stereocenter.
        for index, point in enumerate(achiral.GetConformer().GetPositions()):
            mol.GetConformer().SetAtomPosition(index, tuple(point))
        self.assertEqual(independently_perceived_centers(mol), {})
        self.assertFalse(generation._agdiff_chirality_matches(mol, {1: 'R'})[0])
        stats = generation._agdiff_bond_stats(mol)
        self.assertGreater(stats[0], 0.8)
        self.assertLess(stats[1], 2.0)
        result, _ = self.run_filter([mol], smiles=EZ_DISTINGUISHED)
        self.assertEqual(result, (0, []))

    def test_planar_specified_centers_fail_despite_valid_bond_lengths(self):
        stats = generation._agdiff_bond_stats(self.planar)
        self.assertGreater(stats[0], 0.8)
        self.assertLess(stats[1], 2.0)
        self.assertFalse(generation._agdiff_chirality_matches(self.planar, self.specified)[0])
        result, _ = self.run_filter([self.planar])
        self.assertEqual(result, (0, []))

    def test_rotated_planar_geometry_is_rejected(self):
        planar = Chem.Mol(self.planar)
        conf = planar.GetConformer()
        for index, (x, y, _) in enumerate(conf.GetPositions()):
            conf.SetAtomPosition(index, (x, y / math.sqrt(2), y / math.sqrt(2)))
        conf.Set3D(True)
        self.assertFalse(generation._agdiff_chirality_matches(planar, self.specified)[0])

    def test_nonfinite_coordinates_fail_geometry_and_chirality(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                mol = Chem.Mol(self.all_s)
                conf = mol.GetConformer()
                index = mol.GetNumAtoms() - 1  # NaN is not the first bond distance.
                point = conf.GetAtomPosition(index)
                conf.SetAtomPosition(index, (value, point.y, point.z))
                self.assertIsNone(generation._agdiff_bond_stats(mol))
                self.assertFalse(generation._agdiff_chirality_matches(mol, self.specified)[0])

    def test_nonfinite_unbonded_atom_is_also_rejected(self):
        mol = Chem.CombineMols(self.all_s, Chem.MolFromSmiles('[Na+]'))
        mol.GetConformer().SetAtomPosition(mol.GetNumAtoms() - 1, (float('nan'), 0, 0))
        self.assertIsNone(generation._agdiff_bond_stats(mol))
        self.assertFalse(generation._agdiff_chirality_matches(mol, self.specified)[0])

    def test_nonfinite_distance_from_finite_coordinates_is_rejected(self):
        mol = Chem.Mol(self.all_s)
        mol.GetConformer().SetAtomPosition(0, (1e308, 0, 0))
        mol.GetConformer().SetAtomPosition(1, (-1e308, 0, 0))
        self.assertIsNone(generation._agdiff_bond_stats(mol))

    def test_no_conformer_fails_instead_of_trusting_template(self):
        mol = Chem.Mol(self.all_s)
        mol.RemoveAllConformers()
        self.assertIsNone(generation._agdiff_bond_stats(mol))
        self.assertFalse(generation._agdiff_chirality_matches(mol, self.specified)[0])

    def test_perception_exception_fails_closed(self):
        # Fault injection is needed to exercise the external perception failure.
        with mock.patch.object(Chem, 'AssignStereochemistryFrom3D',
                               side_effect=RuntimeError('forced perception failure')):
            self.assertFalse(generation._agdiff_chirality_matches(self.all_s, self.specified)[0])
            result, _ = self.run_filter([self.all_s])
        self.assertEqual(result, (0, []))

    def test_empty_perception_result_does_not_fall_back_to_template(self):
        # A successful but empty perception result must not resurrect graph tags.
        with mock.patch.object(Chem, 'FindMolChiralCenters', return_value=[]):
            self.assertFalse(generation._agdiff_chirality_matches(self.all_s, self.specified)[0])

    def test_perception_failure_is_not_treated_as_achiral(self):
        with mock.patch.object(Chem, 'AssignStereochemistryFrom3D',
                               side_effect=RuntimeError('forced perception failure')):
            self.assertFalse(generation._agdiff_chirality_matches(embed('CCO'), {})[0])

    def test_missing_expected_atom_fails(self):
        self.assertFalse(generation._agdiff_chirality_matches(embed('CCO'), self.specified)[0])

    def test_partly_missing_expected_centers_fail(self):
        expected = dict(self.specified)
        expected[self.all_s.GetNumAtoms() + 1] = 'S'
        self.assertFalse(generation._agdiff_chirality_matches(self.all_s, expected)[0])

    def test_unknown_expected_label_cannot_match_unassigned_geometry(self):
        self.assertFalse(generation._agdiff_chirality_matches(self.planar, {1: '?'})[0])

    def test_real_3d_coordinates_override_false_2d_flag(self):
        mol = Chem.Mol(self.all_s)
        mol.GetConformer().Set3D(False)
        ok, centers, reason = generation._agdiff_chirality_matches(mol, self.specified)
        self.assertTrue(ok)
        self.assertEqual(centers, self.specified)
        self.assertEqual(reason, 'specified_centers_match')
        self.assertFalse(mol.GetConformer().Is3D(), 'Perception must not mutate the input')

    def test_unspecified_center_policy_is_preserved(self):
        centers, specified = generation._agdiff_target_chiral_centers(PARTLY_SPECIFIED)
        self.assertEqual(centers, {1: 'S', 6: '?'})
        self.assertEqual(specified, {1: 'S'})
        partly_opposite = embed('N[C@@H](C)C(=O)N[C@H](C)C(=O)O')
        self.assertEqual(independently_perceived_centers(partly_opposite), {1: 'S', 6: 'R'})
        result, saved = self.run_filter([partly_opposite], smiles=PARTLY_SPECIFIED)
        self.assertEqual(result, (1, [0]))
        self.assertEqual(independently_perceived_centers(saved[0]), {1: 'S', 6: 'R'})

    def test_achiral_molecule_with_valid_geometry_is_accepted(self):
        mol = embed('CCO')
        self.assertEqual(independently_perceived_centers(mol), {})
        result, _ = self.run_filter([mol], smiles='CCO')
        self.assertEqual(result, (1, [0]))

    def test_achiral_molecule_without_conformer_is_not_accepted(self):
        self.assertFalse(generation._agdiff_chirality_matches(Chem.MolFromSmiles('CCO'), {})[0])

    def test_bond_length_failure_is_rejected(self):
        for length in (0.1, 4.0):
            with self.subTest(length=length):
                mol = Chem.Mol(self.all_s)
                conf = mol.GetConformer()
                index = mol.GetNumAtoms() - 1
                neighbor = mol.GetAtomWithIdx(index).GetNeighbors()[0].GetIdx()
                point = conf.GetAtomPosition(neighbor)
                conf.SetAtomPosition(index, (point.x + length, point.y, point.z))
                result, _ = self.run_filter([mol])
                self.assertEqual(result, (0, []))

    def test_reflection_requires_successful_fresh_recheck(self):
        original = Chem.AssignStereochemistryFrom3D
        calls = 0

        def fail_on_recheck(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('forced post-reflection failure')
            return original(*args, **kwargs)

        with mock.patch.object(Chem, 'AssignStereochemistryFrom3D',
                               side_effect=fail_on_recheck):
            result, _ = self.run_filter([self.all_r])
        self.assertEqual(calls, 2)
        self.assertEqual(result, (0, []))

    def test_filter_caps_at_requested_target_and_preserves_source_indices(self):
        result, saved = self.run_filter(
            [self.mixed, self.all_s, self.all_r, self.all_s], num_needed=2)
        self.assertEqual(result, (2, [1, 2]))
        self.assertEqual([mol.GetIntProp('agdiff_source_final_index') for mol in saved], [1, 2])
        for mol in saved:
            self.assertEqual(independently_perceived_centers(mol), self.specified)

    def test_zero_requested_target_writes_no_conformers(self):
        result, _ = self.run_filter([self.all_s], num_needed=0)
        self.assertEqual(result, (0, []))


if __name__ == '__main__':
    unittest.main()
