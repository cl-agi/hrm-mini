import csv
from pathlib import Path
import tempfile
import unittest
from hrm_qwen.cot.ratings import match_ratings


class RatingsTest(unittest.TestCase):
    def test_join_keeps_selected_order_and_checks_ambiguity(self):
        rows=[dict(question='02',answer='12'),dict(question='10',answer='12')]
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.csv'
            source.write_text('question,answer,rating\n1.,12,7\n.2,12,4\n')
            self.assertEqual(match_ratings(rows,source),[4,7])
            with source.open('a') as stream:stream.write('.2,12,9\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):match_ratings(rows,source)
            source.write_text('question,answer,rating\n.2,12,4\n')
            with self.assertRaisesRegex(ValueError,'missing'):match_ratings(rows,source)
