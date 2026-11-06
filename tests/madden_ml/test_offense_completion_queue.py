import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_offense_completion_queue as module

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source' if (ROOT / 'source').exists() else ROOT.parents[1] / 'cfb_coach' / 'data' / 'madden27'


class InstalledCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((SOURCE / 'installed_playbook_snapshot.json').read_text())
        cls.source = json.loads((SOURCE / 'installed_source_catalog_reference.json').read_text())
        cls.rows = module.build(cls.snapshot, cls.source, {}, {})

    def test_93_exact_records(self):
        self.assertEqual(len(self.rows), 93)
        self.assertEqual(len({(r['formation'], r['play']) for r in self.rows}), 93)

    def test_all_five_formation_counts(self):
        expected = {'Gun 5WR Tight':15, 'Gun Doubles Clamp Stack':27,
                    'Gun Trips X Nasty':21, 'Pistol Deuce Close':12,'Pistol Trips':18}
        for k, v in expected.items():
            self.assertEqual(sum(r['formation']==k for r in self.rows), v)

    def test_11_players_in_every_formation(self):
        for f, p in module.PERSONNEL.items():
            self.assertEqual(sum(p[k] for k in ('WR','TE','HB','QB','OL')), 11, f)

    def test_empty_has_no_halfback(self):
        for r in self.rows:
            if r['formation']=='Gun 5WR Tight':
                self.assertEqual(r['personnel'],'5WR/0TE/0HB')
                self.assertNotIn('HB',r['expected_skill_slots'])

    def test_pistol_deuce_two_tight_ends(self):
        for r in self.rows:
            if r['formation']=='Pistol Deuce Close':
                self.assertIn('TE2', r['expected_skill_slots'])

    def test_no_unauthorized_exact_claims(self):
        for r in self.rows:
            self.assertEqual(r['play_art_capture'],'MISSING')
            self.assertEqual(r['player_assignments_verified'],0)
            self.assertFalse(r['ready_for_training'])

    def test_source_catalog_mismatch_fails(self):
        bad = copy.deepcopy(self.source)
        bad['books']['offense']['Buccaneers']['formations']['Pistol Trips']['plays'].clear()
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            module.build(self.snapshot,bad,{}, {})

    def test_fingerprint_mismatch_fails(self):
        bad=copy.deepcopy(self.snapshot)
        bad['sides']['offense']['inventory_fingerprint']='OFF-NOTREAL'
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            module.build(bad,self.source,{}, {})

    def test_eligible_frequency_priorities_do_not_change_assignment_truth(self):
        t=('Gun 5WR Tight','Mesh')
        rows=module.build(self.snapshot,self.source,{t:10},{t:7})
        self.assertEqual(rows[0]['formation'],'Gun 5WR Tight')
        self.assertEqual(rows[0]['verified_execution_snap_count'],7)
        self.assertEqual(rows[0]['player_assignments_verified'],0)
        self.assertFalse(rows[0]['ready_for_training'])


if __name__ == '__main__': unittest.main()