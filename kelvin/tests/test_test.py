import unittest


class TestTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-14

    def test_framework(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
