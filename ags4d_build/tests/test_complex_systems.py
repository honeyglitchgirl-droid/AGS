import numpy as np
from ags_sci.complex_systems import *

def test_gr_constant_metric_christoffel_zero():
    g=np.diag([-1.,1,1,1]); assert np.max(np.abs(christoffel_symbols(g,np.zeros((4,4,4)))))==0

def test_qft_action_nonnegative(): assert scalar_field_action(np.ones((4,4)))>0

def test_chaos_logistic_bounds():
    x=logistic_map(.2,3.9,100); assert np.all((x>=0)&(x<=1))

def test_burgers_zero(): assert np.allclose(burgers_rhs(np.ones(32),1/32),0)
def test_entropy_uniform(): assert abs(entropy(np.ones(4))-np.log(4))<1e-14
def test_ricker_center(): assert abs(ricker(np.array([0.]),5)[0]-1)<1e-14
def test_lv_equilibrium(): assert np.allclose(lotka_volterra([10,10],alpha=1,beta=.1,delta=.1,gamma=1),0)
def test_plasma_positive(): assert plasma_frequency(1e18)>0
def test_ml_step_improves_simple_fit():
    X=np.array([[1.],[2.],[3.]]); y=2*X[:,0]; w=linear_regression_step(X,y,np.array([0.]),lr=.05); assert abs(w[0])>0
def test_attention_rows_sum_one():
    out,w=scaled_dot_product_attention(np.eye(2),np.eye(2),np.eye(2)); assert np.allclose(w.sum(1),1)
def test_qubit_z(): assert abs(qubit_expectation([1,0],"Z")-1)<1e-14
def test_mi_independent_zero(): assert abs(mutual_information(np.ones((2,2))))<1e-14
def test_gd_moves_to_target(): assert np.allclose(gradient_descent_quadratic(np.eye(2),np.ones(2),np.zeros(2),lr=1),np.ones(2))
def test_fft_constant(): assert np.abs(fft_transform(np.ones(4))[0]-4)<1e-14
def test_metric_inverse(): assert np.allclose(metric_inverse(np.eye(4)),np.eye(4))
def test_euler_sphere(): assert euler_characteristic([1,4,4,1])==0
def test_primes(): assert primes_upto(10)==[2,3,5,7]
def test_category_compose():
    h=compose_functions(lambda x:x+1,lambda x:2*x); assert h(3)==7
def test_ito_zero_dt(): assert ito_euler_step(3,lambda x:x,lambda x:2,0,0)==3
def test_bs_positive(): assert black_scholes_call(100,100,1,.05,.2)>0
def test_pagerank_normalized(): assert abs(pagerank(np.array([[0,1],[1,0]])).sum()-1)<1e-14
def test_nash_prisoners_dilemma():
    A=np.array([[-1,-3],[0,-2]]); B=np.array([[-1,0],[-3,-2]]); assert pure_nash(A,B)==[(1,1)]
def test_string_straight(): assert abs(string_energy(np.array([[0,0],[3,4]]))-5)<1e-14
def test_larmor_positive(): assert larmor_radius(1,2,1,1)==2
def test_scaling_exponent(): assert abs(scaling_exponent([1,2,4],[1,4,16])-2)<1e-14
def test_registry_25(): assert len(REGISTRY)==25
