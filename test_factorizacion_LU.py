import unittest
import numpy as np

from numpy.testing import assert_allclose
from template_funciones import *

class TestLUFactorization(unittest.TestCase):

    def test_matriz_identidad(self):
        A = np.identity(3)
        
        L, U = calculaLU(A)
        
        assert_allclose(L, np.identity(3))
        assert_allclose(U, np.identity(3))
        
        print("test_matriz_identidad - Todo en orden")

    def test_matriz_2x2(self):
        A = np.array([[4.0, 3.0], [6.0, 3.0]])
        
        L_expected = np.array([[1.0, 0.0], [1.5, 1.0]])
        U_expected = np.array([[4.0, 3.0], [0.0, -1.5]])
        L, U = calculaLU(A)
        
        assert_allclose(L, L_expected)
        assert_allclose(U, U_expected)
        assert_allclose(L @ U, A)
        
        print("test_matriz_2x2 - Todo en orden")

    def test_matriz_3x3(self):
        A = np.array([[2.0, 3.0, 1.0], [4.0, 7.0, 2.0], [6.0, 18.0, -1.0]])
        
        L, U = calculaLU(A)
        
        assert_allclose(L @ U, A)

        print("test_matriz_3x3 - Todo en orden")

    def test_matriz_6x6(self):
        L = np.array([[1,0,0,0,0,0], [5,1,0,0,0,0], [8,5,1,0,0,0], [-2,3,2,1,0,0], [4,-6,-2,1,1,0], [2,-4,-3,-2,4,1]])
        U = np.array([[8,5,3,1,5,20], [0,1,5,23,5,4], [0,0,1,5,2,3], [0,0,0,2,35,20], [0,0,0,0,20,20], [0,0,0,0,0,101]])

        L1, U1 = calculaLU(L @ U)

        assert np.allclose(L, L1)
        assert np.allclose(U, U1)

        print("test_matriz_6x6 - Todo en orden")

    def test_zero_pivot(self):
        A = np.array([[0.0, 2.0], [1.0, 3.0]])
        
        with self.assertRaises(ZeroDivisionError):
            calculaLU(A)

        print("test_zero_pivot - Todo en orden")

    def test_recomposicion_de_matriz(self):
        np.random.seed(0)
        A = np.random.rand(5, 5)
        
        L, U = calculaLU(A)
        
        assert_allclose(L @ U, A)

        print("test_recomposicion_de_matriz - Todo en orden")


if __name__ == '__main__':
    unittest.main()