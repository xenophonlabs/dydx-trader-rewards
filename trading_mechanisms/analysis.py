import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import random

from utils import *

PATH = __file__.replace("analysis.py", "images/")

def plot1(R, p, alpha, open_interest):
    profit, profit_fast = setup(R, p, analysis=True)
    d_mkt, f_mkt = find_equilibrium(open_interest, n=1000, R=R, p=p, alpha=alpha)
    mkt_score = cur_mkt_score(d_mkt, f_mkt)
    
    k = 0 
    f_k = f_mkt[k]
    d_k = d_mkt[k]
    T_k = mkt_score - f_k**0.7 * d_k**0.3
    profit_k = profit.evalf(subs={'f':f_k, 'd':d_k, 'T':T_k}) 
    num_points = 1000
    fees = np.linspace(0, 2*max(f_mkt), num=num_points)
    temp_d_mkt = np.ones(num_points) * d_mkt[k] 
    temp_T_mkt = np.array([])
    for fee in fees:
        new_f_mkt = f_mkt.copy()
        new_f_mkt[k] = fee
        mkt_score = cur_mkt_score(d_mkt, new_f_mkt)
        T_mkt = mkt_score - new_f_mkt**0.7 * d_mkt**0.3
        temp_T_mkt = np.append(temp_T_mkt, T_mkt[k])
    
    y = profit_fast(fees, temp_d_mkt, temp_T_mkt)
    for i in range(num_points):
        if y[i] < 0:
            break
    
    plt.plot(fees[:i], y[:i])
    plt.scatter([f_k], [profit_k])
    plt.xlabel("Fees")
    plt.ylabel("Profit")
    plt.title("Individual Fees vs Profit")
    plt.savefig(PATH + "plot1.png")
    return

def plot2(R, p, alpha, open_interest):
    fig, axs = plt.subplots(1,2,figsize=(10,5))

    # Vary large n
    for n in [1000, 2000, 3000, 4000, 5000]:
        d_mkt, f_mkt = find_equilibrium(open_interest, n=n, R=R, p=p, alpha=alpha)
        result = sorted(list(zip(d_mkt, f_mkt)), key=lambda x : x[1]) 
        axs[0].plot(*zip(*result), label=f"n={n}")
    axs[0].legend()
    axs[0].set_xlabel("Open Interest")
    axs[0].set_ylabel("Fees")
    axs[0].set_title("Fees to Open Interest for Large n", y=1.05)
    
    # Vary small n
    for n in [2, 4, 6, 8, 10]:
        d_mkt, f_mkt = find_equilibrium(open_interest, n=n, R=R, p=p, alpha=alpha)
        result = sorted(list(zip(d_mkt, f_mkt)), key=lambda x : x[1]) 
        axs[1].plot(*zip(*result), label=f"n={n}")
    axs[1].legend()
    axs[1].set_xlabel("Open Interest")
    axs[1].set_ylabel("Fees")
    axs[1].set_title("Fees to Open Interest for Small n", y=1.05)
    
    fig.savefig(PATH + "plot2.png")
    return    

def plot3(R, p, alpha, open_interest):
    fig, axs = plt.subplots(1,2,figsize=(10,5),sharey=True)
    func = np.vectorize(lambda x: 0.7*R*p*x/open_interest)

    # Without whales
    d_mkt, f_mkt = find_equilibrium(open_interest, n=1000, R=R, p=p, alpha=alpha)
    d_mkt, f_mkt = zip(*sorted(list(zip(d_mkt, f_mkt)), key=lambda x : x[1]) )
    y = func(d_mkt)
    err = (f_mkt - y)/f_mkt
    axs[0].scatter(d_mkt, err, color='red')
    axs[0].set_xlabel("Open Interest")
    axs[0].set_ylabel("Error")
    axs[0].set_title("Error without Whales", y=1.05)
    
    # With whales
    d_mkt, f_mkt = find_equilibrium(open_interest, \
        n=1000, R=R, p=p, alpha=alpha, num_whales=10, whale_alpha=100)
    d_mkt, f_mkt = zip(*sorted(list(zip(d_mkt, f_mkt)), key=lambda x : x[1]) )
    y = func(d_mkt)
    err = (f_mkt - y)/f_mkt
    axs[1].scatter(d_mkt, err, color='red')
    axs[1].set_xlabel("Open Interest")
    axs[1].set_title("Error with Whales", y=1.05)
    
    fig.savefig(PATH + "plot3.png")
    return

def plot4(R, p, alpha, open_interest):
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    
    sums = []
    whale_alphas = [10_000, 1000, 100, 10, 1]
    for whale_alpha in whale_alphas:
        _, f_mkt = find_equilibrium(open_interest, \
            n=1000, R=R, p=p, alpha=alpha, num_whales=10, whale_alpha=whale_alpha)
        sums.append(sum(f_mkt))
    
    ax.bar([str(x) for x in whale_alphas], sums)
    ax.axhline(y=0.7*R*p, color='red', linestyle='-')
    ax.set_ylabel("Sum of Fees")
    ax.set_xlabel("Whale alpha")
    ax.set_title("Sum of Fees for Varying Whale Sizes")
    
    fig.savefig(PATH + "plot4.png")
    return

def plot5(R, p, alpha, open_interest):
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    
    _, profit_fast = setup(R, p, analysis=True)

    # Without whales
    d_mkt, f_mkt = find_equilibrium(open_interest, n=1000, R=R, p=p, alpha=alpha)
    mkt_score = cur_mkt_score(d_mkt, f_mkt)
    T_mkt = mkt_score - f_mkt**0.7 * d_mkt**0.3
    profits = profit_fast(f_mkt, d_mkt, T_mkt)
    results = sorted(list(zip(d_mkt, profits / f_mkt)), key=lambda x : x[1])

    axs[0].scatter(*zip(*results))
    axs[0].set_xlabel("Open Interest")
    axs[0].set_ylabel("Non-Whale Profit")
    axs[0].set_title("Profit vs Open Interest without Whales")

    # With whales
    d_mkt, f_mkt = find_equilibrium(open_interest, \
        n=1000, R=R, p=p, alpha=alpha, num_whales=10, whale_alpha=100)
    mkt_score = cur_mkt_score(d_mkt, f_mkt)
    T_mkt = mkt_score - f_mkt**0.7 * d_mkt**0.3
    profits = profit_fast(f_mkt, d_mkt, T_mkt) 
    results = sorted(list(zip(d_mkt, profits / f_mkt)), key=lambda x : x[1])

    axs[1].scatter(*zip(*results), color='red')
    axs[1].set_ylabel("Whale Profit")
    axs[1].set_xlabel("Open Interest")
    axs[1].set_title("Profit vs Open Interest with Whales")
    
    fig.savefig(PATH + "plot5.png")
    return

def plot6(R, p, alpha, open_interest):
    fig, axs = plt.subplots(1,1,figsize=(10,5))

    for G in [2000, 4000, 6000, 8000]:
        d_mkt, f_mkt, g_mkt = find_equilibrium_stk(open_interest, G=G, n=1000, R=R, p=p, alpha=alpha, num_whales=10, whale_alpha=100)
        results = sorted(list(zip(d_mkt**0.28 * g_mkt**0.05, f_mkt**0.33)), key=lambda x : x[1])
        axs.plot(*zip(*results), label=f"G={G}")
    
    axs.legend()
    axs.set_xlabel("d_k^0.28 * g_k^0.05")
    axs.set_ylabel("f_k^0.33")
    axs.set_title("Distribution of Fees to Open Interest for varying G")

    fig.savefig(PATH + "plot6.png")
    return

def plot7():
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    df = pd.read_csv(__file__.replace("analysis.py", "data/historical.csv"))

    axs[0].bar(df['epoch'], df['fees'], color='red')
    axs[0].set_title('Fees Paid')
    axs[0].set_ylabel('Fees Paid')
    axs[0].set_xlabel('epoch')
    axs[1].bar(df['epoch'], df['openInterest'], color='blue')
    axs[1].set_title('Final Open Interest')
    axs[1].set_ylabel('Final Open Interest')
    axs[1].set_xlabel('epoch')
    axs[2].bar(df['epoch'], df['close'], color='green')
    axs[2].set_title('Price')
    axs[2].set_ylabel('Price}')
    axs[2].set_xlabel('epoch')

    fig.savefig(PATH + "plot7.png")
    return

def plot8(alpha):
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))

    df = pd.read_csv(__file__.replace("analysis.py", "data/historical.csv"))
    
    results = dict()
    for i in range(df.shape[0]):
        row = df.iloc[i]
        epoch = row['epoch']
        if epoch > 4:
            results[epoch] = find_equilibrium_stk(row['openInterest'], G=row['stakedDYDX'], \
                 n=row['numTraders'], R=row['totalRewards'], p=row['close'], alpha=alpha, \
                     num_whales=10, whale_alpha=100)
        else:
            results[epoch] = find_equilibrium(row['openInterest'], \
                n=row['numTraders'], R=row['totalRewards'], p=row['close'], alpha=alpha, \
                     num_whales=10, whale_alpha=100)

    df['expected'] = [sum(results[epoch][1]) for epoch in results] # sum of fees
    axs.scatter(df['epoch'], df['fees'], label='fees')
    axs.scatter(df['epoch'], df['expected'], label='expected')
    axs.legend()

    fig.savefig(PATH + "plot8.png")
    return

def main():
    """
    This creates all the tables and plot for our analysis.
    """
    R = 3_835_616
    p = 10
    alpha = 0.01
    open_interest = 150_000_000

    # (1) Individual profit curve with black dot on maximum
    print("Generating plot 1... ")
    plot1(R, p, alpha, open_interest)

    # (2) Side by side fees to open interest for large vs small varying n
    print("Generating plot 2... ")
    plot2(R, p, alpha, open_interest)

    # (3) Error of closed form with and without whales
    print("Generating plot 3... ")
    plot3(R, p, alpha, open_interest)

    # (4) Bar chart sum of fees for increasing whale size
    print("Generating plot 4... ")
    plot4(R, p, alpha, open_interest)

    # (5) Profit vs open interest with and without whales
    print("Generating plot 5... ")
    plot5(R, p, alpha, open_interest)
    
    # (6) Plot normal G and plot where G = 0
    print("Generating plot 6... ")
    plot6(R, p, alpha, open_interest)
    
    # (7) Plot actual data (price, fees, open interest)
    print("Generating plot 7... ")
    plot7()

    # (8) Plot of our predictions vs actual data
    print("Generating plot 8... ")
    plot8(alpha)
    
    return

if __name__ == "__main__":
    main()
