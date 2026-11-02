"""Pure draft/verification invariants, including rejection of a plausible wrong draft."""
import unittest
from engine.runtime.thinker_lookup import prompt_draft, verified_count

class PromptLookupTests(unittest.TestCase):
    def test_no_forced_echo(self):
        self.assertEqual(prompt_draft([1,2,3,4,5],[90,91],8),[])
        draft=prompt_draft([1,2,3,4,5],[1,2],8)
        self.assertEqual(draft,[3,4,5])
        self.assertEqual(verified_count(draft,[99,4,5,6]),1)
    def test_partial_rejection_keeps_target_mismatch(self):
        self.assertEqual(verified_count([3,4,5],[3,77,5,6]),2)
    def test_all_match_includes_verified_bonus(self):
        self.assertEqual(verified_count([3,4],[3,4,8]),3)
    def test_bounded_and_longest_suffix(self):
        self.assertEqual(prompt_draft([1,2,9,1,2,3,4,5],[1,2],2),[3,4])
        self.assertEqual(prompt_draft([1,2,3],[1,2],0),[])
    def test_incomplete_predictions_rejected(self):
        with self.assertRaises(ValueError):verified_count([3,4],[3,4])

if __name__=='__main__':unittest.main()
