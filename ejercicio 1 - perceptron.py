import numpy as np
import unittest


class TestPerceptron(unittest.TestCase):

    def test_and(self):
        and_perceptron = Perceptron([1, 1], -1.5)
        self.assertEqual(Logic(and_perceptron, [0, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 1]), 1)

        and_perceptron = Perceptron([1, 1, 1, 1], -3.5)
        self.assertEqual(Logic(and_perceptron, [0, 0, 0, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 0, 0, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 0, 1, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 0, 1, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 1, 0, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 1, 0, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 1, 1, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [0, 1, 1, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 0, 0, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 0, 0, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 0, 1, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 0, 1, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 1, 0, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 1, 0, 1]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 1, 1, 0]), 0)
        self.assertEqual(Logic(and_perceptron, [1, 1, 1, 1]), 1)

    def test_or(self):
        or_perceptron = Perceptron([1, 1], -0.5)
        self.assertEqual(Logic(or_perceptron, [0, 0]), 0)
        self.assertEqual(Logic(or_perceptron, [0, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 1]), 1)

        or_perceptron = Perceptron([1, 1, 1, 1], -0.5)
        self.assertEqual(Logic(or_perceptron, [0, 0, 0, 0]), 0)
        self.assertEqual(Logic(or_perceptron, [0, 0, 0, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 0, 1, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 0, 1, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 1, 0, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 1, 0, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 1, 1, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [0, 1, 1, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 0, 0, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 0, 0, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 0, 1, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 0, 1, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 1, 0, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 1, 0, 1]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 1, 1, 0]), 1)
        self.assertEqual(Logic(or_perceptron, [1, 1, 1, 1]), 1)

    def test_nand(self):
        nand_perceptron = Perceptron([-1, -1], 1.5)
        self.assertEqual(Logic(nand_perceptron, [0, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 1]), 0)

        nand_perceptron = Perceptron([-1, -1, -1, -1], 3.5)
        self.assertEqual(Logic(nand_perceptron, [0, 0, 0, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 0, 0, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 0, 1, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 0, 1, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 1, 0, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 1, 0, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 1, 1, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [0, 1, 1, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 0, 0, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 0, 0, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 0, 1, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 0, 1, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 1, 0, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 1, 0, 1]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 1, 1, 0]), 1)
        self.assertEqual(Logic(nand_perceptron, [1, 1, 1, 1]), 0)


class Perceptron:
    def __init__(self, listw, bias):
        self.listw = listw
        self.bias = bias


def Logic(perceptron, inp):
    x = np.array(inp)
    w = np.array(perceptron.listw)
    y = np.sum(w * x)
    if y + perceptron.bias <= 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    unittest.main()
