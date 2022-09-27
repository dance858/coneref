#Modified from Enzo Busseti 2017-2019.
import unittest
import numpy as np
import coneref

HOW_MANY_DERIVATIVE_SAMPLES = 1000
HOW_LONG_DERIVATIVE_TEST_STEP = 1e-6


def size_vec(x):
    return 1 if isinstance(x, float) else len(x)


class BaseTestCone(unittest.TestCase):

    """Base class for cones tests."""

    sample_vecs = []
    sample_vec_proj = []
    sample_vecs_are_in = []
    sample_vecs_are_diff = []

    def make_cache(self, n):
        if self.test_cone == "SOC":
            return np.zeros(n)
        elif self.test_cone == "EXP":
            return np.zeros((n))
        elif self.test_cone == "EXP_DUAL":
            return np.zeros((n))
        elif self.test_cone == "PSD":
            eval_cache = np.zeros(np.sum(n))
            evec_cache = np.zeros(np.sum(n**2))
            return [eval_cache, evec_cache]

    def Pi(self, x, cache):
        if self.test_cone == "SOC":
            return coneref.SOC_Pi(x, cache)
        elif self.test_cone == "PSD":
            return coneref.PSD_Pi(x, cache[0], cache[1])
        elif self.test_cone == "EXP":
            return coneref.exp_primal_Pi(x, cache)
        elif self.test_cone == "EXP_DUAL":
            return coneref.exp_dual_Pi(x, cache)
   
    def diff(self, x, delta, cache):
        #print("DIFF FUNCTION")
        if self.test_cone == "SOC":
            diff_operator = coneref.SOC_Pi_diff(x, cache)
            #print("Differentiating SOC")
        elif self.test_cone == "PSD":
            diff_operator = coneref.SDP_Pi_diff(x, cache[0], cache[1])
        elif self.test_cone == "EXP":
            diff_operator = coneref.exp_primal_Pi_diff(x, cache)
        elif self.test_cone == "EXP_DUAL":
            diff_operator = coneref.exp_dual_Pi_diff(x, cache)    
        return diff_operator.matvec(delta)

    def test_contains(self):
        for x, isin in zip(self.sample_vecs, self.sample_vecs_are_in):
             cache = self.make_cache(len(x))
             res = self.Pi(x, cache)
             Pix = res
             self.assertTrue(np.alltrue(Pix == x) == isin)

    def test_proj(self):
        for x, proj_x in zip(self.sample_vecs, self.sample_vec_proj):
             cache = self.make_cache(len(x))
             Pix = self.Pi(x, cache)
             if not np.allclose(Pix, proj_x):
                print("Pix:", Pix)
                print("proj_x:", proj_x)
                print("x:", x)
                print(" ")
             self.assertTrue(np.allclose(Pix, proj_x))
        


    def test_derivative_random(self):
        for x, isdiff in zip(self.sample_vecs,
                             self.sample_vecs_are_diff):
            x = np.random.randn(len(x))
            cache = self.make_cache(len(x))
            proj_x = self.Pi(x, cache)

            # Due to Pybind11-properties we must (for the test) fill the cache manually.
            if self.test_cone == "SOC":
                cache = coneref.SOC_Pi(x, cache)
            elif self.test_cone == "PSD":
                X = coneref.matrix_from_lower_triangular(x)
                evals, evecs = np.linalg.eigh(X)
                cache[0] = evals
                cache[1] = evecs.flatten('F')
            elif self.test_cone == "EXP":
                proj = coneref.exp_primal_Pi(x, cache)
                cache[0] = proj[0]
                cache[1] = proj[1]
                cache[2] = proj[2] - x[2]
            elif self.test_cone == "EXP_DUAL":
                # Filling this cache manually is complicated. We use a hack.
                dict_of_cones = {'ed': 1}
                q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
                  = coneref.make_prod_cone_cache(dict_of_cones)
                cones = coneref.parse_cone_dict(dict_of_cones)
                _cones_parsed = coneref.parse_cone_dict_cpp(cones)
                return_info = coneref.prod_cone_Pi_return_cache(x, _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
                cache = return_info[5]
            
            for i in range(HOW_MANY_DERIVATIVE_SAMPLES):
                delta = np.random.randn(
                    size_vec(x)) * HOW_LONG_DERIVATIVE_TEST_STEP*0.1
                # new_cache is not used later.
                new_cache = self.make_cache(len(x))
                proj_x_plus_delta = self.Pi(x + delta, new_cache)
 
                dproj_x = self.diff(x, delta, cache)
                if not np.allclose(proj_x + dproj_x, proj_x_plus_delta, atol = 1e-5):
                    print('x:', x)
                    print('Pi x:', proj_x)
                    print('delta:')
                    print(delta)
                    print('Pi (x + delta) - Pi(x):')
                    print(proj_x_plus_delta - proj_x)
                    print('DPi delta:')
                    print(dproj_x)
                    print("proj_x + dproj_x:")
                    print(proj_x + dproj_x)
                    print("proj_x_plus_delta:")
                    print(proj_x_plus_delta)

                self.assertTrue(np.allclose(
                    proj_x + dproj_x,
                    proj_x_plus_delta, atol = 1e-5))


class TestExpPri(BaseTestCone):

    test_cone = "EXP"
    sample_vecs = [np.array([0., 0., 0.]),
                   np.array([-10., -10., -10.]),
                   np.array([10., 10., 10.]),
                   np.array([1., 2., 3.]),
                   np.array([100., 2., 300.]),
                   np.array([-1., -2., -3.]),
                   np.array([-10., -10., 10.]),
                   np.array([1., -1.,  1.])]
                   #np.array([0.08755124, -1.22543552, 0.84436298])]
    sample_vec_proj = [np.array([0., 0., 0.]),
                       np.array([-10., 0., 0.]),
                       np.array([4.26306172,  7.51672777, 13.25366605]),
                       np.array([0.8899428, 1.94041882, 3.06957225]),
                       np.array([73.77502858,  33.51053837, 302.90131756]),
                       np.array([-1., 0., 0.]),
                       np.array([-10., 0., 10.]),
                       np.array([0.22972088, 0.09487128, 1.06839895])]
                       #np.array([3.88378507e-06, 2.58963810e-07, 0.84436298])]  
    sample_vecs_are_in = [True, False, False,
                          False, False, False, False]#, False]
    sample_vecs_are_diff = [False, True, True, True, True, True, True]#, True]


class TestExpDua(BaseTestCone):

    test_cone = "EXP_DUAL"
    sample_vecs = [np.array([0., 0., 0.]),
                   np.array([-1., 1., 100.]),
                   np.array([1., 1., 100.]),
                   np.array([-1., -2., -3.])]
    sample_vec_proj = [np.array([0., 0., 0.]),
                       np.array([-1., 1., 100.]),
                       np.array([0., 1., 100.]),
                       np.array([-0.1100572, -0.05958119,  0.06957226])]
    sample_vecs_are_in = [True, True, False, False]
    sample_vecs_are_diff = [False, True, True, True]


class TestSecondOrder(BaseTestCone):

    test_cone = "SOC"
    sample_vecs = [np.array([1., 0., 0.]),
                   np.array([1., 2., 2.]),
                   np.array([-10., 2., 2.]),
                   np.array([-2 * np.sqrt(2), 2., 2.]),
                   np.array([-1., 2., 2.]),
                   np.array([0., 1.]),
                   np.array([.5, -.5])]
    sample_vec_proj = [np.array([1., 0., 0.]),
                       [(2 * np.sqrt(2) + 1) / (2),
                        (2 * np.sqrt(2) + 1) / (2 * np.sqrt(2)),
                        (2 * np.sqrt(2) + 1) / (2 * np.sqrt(2))],
                       np.array([0, 0, 0]),
                       np.array([0, 0, 0]),
                       np.array([0.9142135623730951,
                                 0.6464466094067263,
                                 0.6464466094067263
                                 ]),
                       np.array([.5, .5]),
                       np.array([.5, -.5])]
    sample_vecs_are_in = [True, False, False, False, False, False, True]
    sample_vecs_are_diff = [True, True, True, False, True, True, False]


class TestProduct(BaseTestCone):

    test_cone = "prod_cone"

    def test_baseProduct(self):

        # Test 1 and 2
        dict_of_cones = {'l': 3}
        q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
         = coneref.make_prod_cone_cache(dict_of_cones)
        _cones = coneref.parse_cone_dict(dict_of_cones)
        _cones_parsed = coneref.parse_cone_dict_cpp(_cones)
        Pix = coneref.prod_cone_Pi(np.arange(3.), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.alltrue(Pix == np.arange(3.)))
        Pix = coneref.prod_cone_Pi(np.array([1., -1., -1.]), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.alltrue(Pix == [1, 0, 0]))

        # Test 3 and 4
        dict_of_cones = {'l': 2, 's': [1]}
        q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
         = coneref.make_prod_cone_cache(dict_of_cones)
        _cones = coneref.parse_cone_dict(dict_of_cones)
        _cones_parsed = coneref.parse_cone_dict_cpp(_cones)
        Pix = coneref.prod_cone_Pi(np.arange(3.), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.alltrue(Pix == range(3)))
        Pix =  coneref.prod_cone_Pi(np.array([1., -1., -1.]), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.alltrue(Pix  == [1, 0, 0]))

        # Test 5
        dict_of_cones = {'s': [2, 1]}
        q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
         = coneref.make_prod_cone_cache(dict_of_cones)
        _cones = coneref.parse_cone_dict(dict_of_cones)
        _cones_parsed = coneref.parse_cone_dict_cpp(_cones)
        Pix = coneref.prod_cone_Pi(np.arange(4.), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.allclose(Pix - np.array(
              [0.20412415, 0.90824829, 2.02062073, 3]), 0.))
        Pix = coneref.prod_cone_Pi(np.array([1, -20., 1, -1]), _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
        self.assertTrue(np.allclose(Pix, np.array([7.57106781, -10.70710678,
                                                    7.57106781,   0.])))

    def test_deriv_Product(self):
        print(" ")
        dict_of_cones = {'z': 2, 'l': 20,  'q': [3, 4],
                    's': [3, 4, 5], 'ep': 10, 'ed': 10}
        m = 22 + 7 + (6 + 10 + 15) + 30 + 30
        q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
         = coneref.make_prod_cone_cache(dict_of_cones)
        _cones = coneref.parse_cone_dict(dict_of_cones)
        _cones_parsed = coneref.parse_cone_dict_cpp(_cones)

        for j in range(100):
            x = np.random.randn(m)
            q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
             = coneref.make_prod_cone_cache(dict_of_cones)

            return_info = coneref.prod_cone_Pi_return_cache(x, _cones_parsed, q_cache, eval_cache, evec_cache, ep_cache, ed_cache)
            proj_x, q_cache, eval_cache, evec_cache, ep_cache, ed_cache = return_info[0], return_info[1], return_info[2], return_info[3], return_info[4], return_info[5]
            diff_operator = coneref.prod_cone_Pi_diff(x, _cones_parsed, q_cache, eval_cache, 
                                 evec_cache, ep_cache, ed_cache)
            for i in range(HOW_MANY_DERIVATIVE_SAMPLES):
                delta = np.random.randn(size_vec(x)) * \
                    HOW_LONG_DERIVATIVE_TEST_STEP
                # print('x + delta:', x + delta)
                q_cache_new, eval_cache_new, evec_cache_new, ep_cache_new, ed_cache_new \
                    = coneref.make_prod_cone_cache(dict_of_cones)
                proj_x_plus_delta = coneref.prod_cone_Pi(x + delta, _cones_parsed, q_cache_new, eval_cache_new, evec_cache_new, ep_cache_new, ed_cache_new)
                dproj_x = diff_operator.matvec(delta)

                if not np.allclose(proj_x + dproj_x, proj_x_plus_delta, atol=1e-5):
                    print(dict_of_cones)
                    print('x:', x)
                    print('Pi (x):', proj_x)
                    # print(delta)
                    print('Pi (x + delta) - Pi(x):')
                    print(proj_x_plus_delta - proj_x)
                    print('DPi delta:')
                    print(dproj_x)

                    print('error:')
                    print(proj_x + dproj_x -
                          proj_x_plus_delta)

                self.assertTrue(np.allclose(
                    proj_x + dproj_x,
                    proj_x_plus_delta, atol=1e-5))


class TestSemiDefinite(BaseTestCone):

    test_cone = "PSD"
    sample_vecs = [np.array([2, 0, 0, 2, 0, 2.]), np.array([1.]), np.array([-1.]),
                   np.array([10, 20., 10.]),
                   np.array([10, 0., -3.]),
                   np.array([10, 20., 0., 10, 0., 10]),
                   np.array([1, 20., 30., 4, 50., 6]),
                   np.array([1, 20., 30., 4, 50., 6, 200., 20., 1., 0.])]
    sample_vec_proj = [[2, 0, 0, 2, 0, 2], [1.], [0.],
                       [[12.07106781, 17.07106781, 12.07106781]],
                       np.array([10, 0., 0.]),
                       np.array([12.07106781, 17.07106781,
                                 0., 12.07106781, 0., 10.]),
                       np.array([10.11931299, 19.85794691, 21.57712079,
                                 19.48442822, 29.94069045,
                                 23.00413782]),
                       np.array([10.52224268,  13.74405405,  21.782617,  10.28175521,
                                 99.29457317,   5.30953205, 117.32861549,  23.76075308,
                                 2.54829623,  69.3944742])]
    sample_vecs_are_in = [True, True, False, False, False, False, False, False]
    sample_vecs_are_diff = [True, True, True, True, True, True, True, True]

if __name__ == '__main__':
    unittest.main()
    
