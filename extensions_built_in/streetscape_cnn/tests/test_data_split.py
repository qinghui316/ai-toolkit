import unittest
from collections import Counter
from extensions_built_in.streetscape_cnn.data import split_candidates, CELLS


def candidates():
    result = []
    for number in range(1, 9):
        code = f'T{number:02d}'
        for index in range(60):
            result.append({'image_id': f'CS_{code}_{index + 1:03d}', 'archetype_code': code,
                           'source_origin': 'original160' if index < 20 else 'added240',
                           'source_batch': f'batch{(index - 20) // 4}' if index >= 20 else 'old',
                           'contrast_cell': sorted(CELLS)[(index - 20) % 4] if index >= 20 else '',
                           'cnn_data_role': 'historical'})
    return result


class DataSplitTests(unittest.TestCase):
    def test_exact_balanced_grouped_reproducible_split(self):
        rows = candidates()
        split = split_candidates(rows)
        self.assertEqual(split, split_candidates(list(reversed(rows))))
        self.assertEqual(Counter(row['lora_data_role'] for row in split),
                         {'train': 384, 'validation': 48, 'test': 48})
        for number in range(1, 9):
            self.assertEqual(Counter(r['lora_data_role'] for r in split if r['archetype_code'] == f'T{number:02d}'),
                             {'train': 48, 'validation': 6, 'test': 6})
        memberships = {}
        for row in split:
            memberships.setdefault(row['split_group'], set()).add(row['lora_data_role'])
        self.assertTrue(all(len(v) == 1 for v in memberships.values()))
        self.assertNotIn('lora_data_role', rows[0])

    def test_missing_duplicate_and_bad_contrast_rejected(self):
        rows = candidates()
        with self.assertRaises(ValueError):
            split_candidates(rows[:-1])
        rows[-1]['image_id'] = rows[0]['image_id']
        with self.assertRaises(ValueError):
            split_candidates(rows)
        rows = candidates()
        rows[20]['contrast_cell'] = 'unknown'
        with self.assertRaises(ValueError):
            split_candidates(rows)

    def test_confirmed_link_constraint_is_never_silently_broken(self):
        rows = candidates()
        existing = split_candidates(rows)
        selected = [next(r for r in existing if r['source_origin'] == 'original160' and r['lora_data_role'] == role)
                    for role in ['train', 'test']]
        ids = {r['image_id'] for r in selected}
        for row in rows:
            if row['image_id'] in ids:
                row['verified_duplicate_group'] = 'confirmed-link'
        with self.assertRaises(ValueError):
            split_candidates(rows)
