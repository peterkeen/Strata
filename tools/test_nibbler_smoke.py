import unittest

from tools.nibbler_smoke import correct_answer


class QualityAnswers(unittest.TestCase):
    def test_numbers(self):
        self.assertTrue(correct_answer('{"answer":1}', 1))
        self.assertTrue(correct_answer('{"answer":1.0}', 1))
        self.assertFalse(correct_answer('{"answer":"1"}', 1))

    def test_booleans_are_not_numbers(self):
        self.assertFalse(correct_answer('{"answer":true}', 1))
        self.assertFalse(correct_answer('{"answer":[true,9,25]}', [1, 9, 25]))
        self.assertTrue(correct_answer('{"answer":true}', True))

    def test_lists_and_strings(self):
        self.assertTrue(correct_answer('{"answer":[1,9,25]}', [1, 9, 25]))
        self.assertFalse(correct_answer('{"answer":[1,9]}', [1, 9, 25]))
        self.assertTrue(correct_answer('{"answer":"unknown"}', 'unknown'))

    def test_exact_schema(self):
        for text in ['1', '[1]', '{}', '{"answer":1,"extra":2}']:
            self.assertFalse(correct_answer(text, 1))

    def test_fences_and_malformed(self):
        self.assertTrue(correct_answer('```json\n{"answer":1}\n```', 1))
        for text in ['', '```', 'not JSON', '{"answer":']:
            self.assertFalse(correct_answer(text, 1))


if __name__ == '__main__':
    unittest.main()
