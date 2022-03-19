import sympy as sp
import numpy as np
import random

def dist(v1, v2):
    """
    Root-mean-squared-error between vector 1 and vector 2
    """
    return (sum([(v1[i]-v2[i])**2 for i in range(len(v1))])/len(v1))**(1/2)

def cur_mkt_score(ds, fs):
    """
    Given a bunch of participants' open-interests and fees,
    calculate the total current market score.
    """
    total = 0
    for d, f in zip(ds, fs):
        total += (d**.3) * (f**.7)
    return total

def cur_mkt_score_stk(ds, fs, gs):
    """
    Given a bunch of participants' open-interests and fees,
    calculate the total current market score.
    """
    total = 0
    for d, f, g in zip(ds, fs, gs):
        total += (d**.28) * (f**.67) * (g**0.05)
    return total

def setup(R, p, analysis=False):
    f = sp.Symbol('f')
    d = sp.Symbol('d')
    T = sp.Symbol('T')

    profit = R * p * d**0.3 * f**0.7 / (T + d**0.3 * f**0.7) - f  

    profit_prime = sp.diff(profit, f)
    profit_prime_prime = sp.diff(profit_prime, f)

    profit_prime_fast = sp.lambdify([f, d, T], profit_prime, "numpy")
    profit_prime_prime_fast = sp.lambdify([f, d, T], profit_prime_prime, "numpy")

    if analysis:
        return profit, sp.lambdify([f, d, T], profit, "numpy")
    
    return profit_prime_fast, profit_prime_prime_fast

def setup_stk(R, p):
    f = sp.Symbol('f')
    d = sp.Symbol('d')
    T = sp.Symbol('T')
    g = sp.Symbol('g')

    profit = R * p * d**0.28 * f**0.67 * g**0.05 / (T + d**0.28 * f**0.67 * g**0.05) - f

    profit_prime = sp.diff(profit, f)
    profit_prime_prime = sp.diff(profit_prime, f)

    profit_prime_fast = sp.lambdify([f, d, g, T], profit_prime, "numpy")
    profit_prime_prime_fast = sp.lambdify([f, d, g, T], profit_prime_prime, "numpy")

    return profit_prime_fast, profit_prime_prime_fast

def walk(alpha, d_mkt, f_mkt, profit_prime_fast, profit_prime_prime_fast):
    mkt_score = cur_mkt_score(d_mkt, f_mkt)
    T_mkt = mkt_score - f_mkt**0.7 * d_mkt**0.3

    d1 = profit_prime_fast(f_mkt, d_mkt, T_mkt)
    d2 = profit_prime_prime_fast(f_mkt, d_mkt, T_mkt)
    new_f_mkt = f_mkt - alpha * d1 / d2

    return new_f_mkt

def walk_stk(alpha, d_mkt, f_mkt, g_mkt, profit_prime_fast, profit_prime_prime_fast):
    mkt_score = cur_mkt_score_stk(d_mkt, f_mkt, g_mkt)
    T_mkt = mkt_score - f_mkt**0.67 * d_mkt**0.28 * g_mkt ** 0.05

    d1 = profit_prime_fast(f_mkt, d_mkt, g_mkt, T_mkt)
    d2 = profit_prime_prime_fast(f_mkt, d_mkt, g_mkt, T_mkt)
    new_f_mkt = f_mkt - alpha * d1 / d2

    return new_f_mkt

def get_mkt(total, n, num_whales, whale_alpha, fill=False):
    """
    Randomly distrbute total amount into n buckets according to
    Dirichlet distribution, where whale_alpha determines the alpha
    parameter of the first num_whales trader, other traders default
    to 1.
    """
    whales = np.array([whale_alpha]*num_whales)
    mkt = np.array(np.random.dirichlet(np.append(whales, np.ones(n-num_whales)), 1).ravel() * total) 

    if fill:
        mkt = np.where(mkt < 10, 10, mkt)

    return mkt

def find_equilibrium(D, n=1000, R=3_835_616, p=20, alpha=.01, num_whales=0, whale_alpha=1):
    """
    Warning: If leaning rate is too small or fees vector is initialized at very high amounts, 
    Newton's method can update fees as negative values. This will crash the algorithm. 
    To avoid this, lower the learning rate or instantiate the fees vector at smaller amounts.

    Parameters:

    """
    profit_prime_fast, profit_prime_prime_fast = setup(R, p)

    # d_mkt = np.random.exponential(scale=(D/n), size=n)
    d_mkt = get_mkt(D, n, num_whales, whale_alpha)
    f_mkt = np.random.rand(n)*((1/25) * D)/n

    # simulate_market_optimal_fee_discovery, running until convergence
    rmses = [] # the distances between fees on successive iterations of the algorithm; should tend to 0 as f_mkt convergest
    while (rmses==[]) or (rmses[-1] > 10**-10):
        new_f_mkt = walk(alpha, d_mkt, f_mkt, profit_prime_fast, profit_prime_prime_fast)
        rmses.append(dist(new_f_mkt, f_mkt))
        f_mkt = new_f_mkt
    
    check_equilibrium(d_mkt, f_mkt, profit_prime_fast)
    return d_mkt, f_mkt

def find_equilibrium_stk(D, n=1000, R=3_835_616, p=20, alpha=.01, G=5_000, num_whales=0, whale_alpha=1):
    profit_prime_fast, profit_prime_prime_fast = setup_stk(R, p)

    # d_mkt = np.random.exponential(scale=(D/n), size=n)
    d_mkt = get_mkt(D, n, num_whales, whale_alpha)
    f_mkt = np.random.rand(n)*((1/25) * D)/n
    g_mkt = get_mkt(G, n, 10, 100000)
    
    # simulate_market_optimal_fee_discovery, running until convergence
    rmses = [] # the distances between fees on successive iterations of the algorithm; should tend to 0 as f_mkt convergest
    while (rmses==[]) or (rmses[-1] > 10**-10):
        new_f_mkt = walk_stk(alpha, d_mkt, f_mkt, g_mkt, profit_prime_fast, profit_prime_prime_fast)
        rmses.append(dist(new_f_mkt, f_mkt))
        f_mkt = new_f_mkt
    
    check_equilibrium_stk(d_mkt, f_mkt, g_mkt, profit_prime_fast)
    return d_mkt, f_mkt, g_mkt

def check_equilibrium(d_mkt, f_mkt, profit_prime_fast):
    """
    Verify that evey trader's f_k minimizes their profit curve.
    """
    mkt_score = cur_mkt_score(d_mkt, f_mkt)
    T_mkt = mkt_score - f_mkt**0.7 * d_mkt**0.3
    err = profit_prime_fast(f_mkt, d_mkt, T_mkt)
    if not np.all((err <= 10e-2)):
        raise Exception("Newton's method did not find an equilibrium.")
    return True

def check_equilibrium_stk(d_mkt, f_mkt, g_mkt, profit_prime_fast):
    """
    Verify that evey trader's f_k minimizes their profit curve.
    """
    mkt_score = cur_mkt_score_stk(d_mkt, f_mkt, g_mkt)
    T_mkt = mkt_score - f_mkt**0.67 * d_mkt**0.28 * g_mkt**0.05
    err = profit_prime_fast(f_mkt, d_mkt, g_mkt, T_mkt)
    if not np.all((err <= 10e-2)):
        raise Exception("Newton's method did not find an equilibrium.")
    return True
