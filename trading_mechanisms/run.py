import argparse

import numpy as np

from utils import find_equilibrium_stk

def parse_args():
    parser = argparse.ArgumentParser("Get Nash-equilibrium fees to pay throughout an epoch.")

    # Required params
    parser.add_argument(
        "-D",
        type=float,
        help="Average total epoch open interest. Can be estimated from the metabase open interest dashboard https://metabase.dydx.exchange/public/dashboard/5fa0ea31-27f7-4cd2-8bb0-bc24473ccaa3.",
        required=True,
        # default=1_000_000_000 # this would mean average open interest of $1B
    )

    parser.add_argument(
        "-d",
        type=float,
        help="Average open interest for the trader running the simulation. Can be found on https://trade.dydx.exchange/portfolio/rewards.",
        required=True,
        # default=10_000, # this would mean $10,000 in average open interest
    )

    parser.add_argument(
        "-p",
        type=float,
        help="The price of the token to be distributed as rewards, at the end of the epoch. Can be found on coingecko https://www.coingecko.com/en/coins/dydx.",
        required=True,
        # default=5.5, # this would mean end of epoch price of $5.50
    )

    
    # Non-Required Params
    parser.add_argument(
        "-n",
        type=int,
        help="The number of participants gaming the rewards mechanism. Only change if you know what you're doing.",
        required=False,
        default=8_000,
    )

    parser.add_argument(
        "--num-whales",
        type=float,
        help="The number of market whales in the simulation. Only change if you know what you're doing.",
        required=False,
        default=10,
    )

    parser.add_argument(
        "-R",
        type=float,
        help="The number of tokens distributed as rewards. Only change if you know what you're doing.",
        required=False,
        default=3_835_616,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="The Newton's method learning rate parameter. Only change if you know what you're doing.",
        required=False,
        default=0.01,
    )

    parser.add_argument(
        "--whale-alpha",
        type=float,
        help="The whales' Dirichlet distribution alpha parameter. Only change if you know what you're doing.",
        required=False,
        default=200,
    )

    parser.add_argument(
        "-G",
        type=float,
        help="The amount of DYDX staked by market participants.",
        required=False,
        default=25_000_000,
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        help="The number of trials to run the algorithm. Larger number means longer run time and more precise results.",
        required=False,
        default=1,
    )

    return vars(parser.parse_args())

def _index_closest_oi(d_me, d_mkt):
    """
    d_me: average open interest for the individual
    d_mkt: average open interest for the market participants in our simulation
    """

    distances = [abs(d_me - d_sim) for d_sim in d_mkt]
    return np.argmin(distances)

def estimate_optimal_fees():
    args = parse_args()

    f_opts = []
    for i in range(args["num_trials"]):
        print(f"Beginning trial {i+1} of {args['num_trials']}.")
        d_mkt, f_mkt, g_mkt = find_equilibrium_stk(
            args["D"],
            args["n"],
            args["R"],
            args["p"],
            args["alpha"],
            args["G"],
            args["num_whales"],
            args["whale_alpha"],
        )
        
        closest_oi_index = _index_closest_oi(
            args["d"],
            d_mkt
        )

        f_ratio = f_mkt[closest_oi_index]/d_mkt[closest_oi_index] # ratios of fees to open interest
        f_opts.append(args["d"] * f_ratio)

    avg_opt_fee = np.mean(f_opts)
    std_opt_fee = np.std(f_opts)

    print(f"Total fees to pay this epoch: ${avg_opt_fee:<.2f}.")
    print(f"Standard deviation of optimal fees paid: ${std_opt_fee:<.2f}")

if __name__ == "__main__":
    estimate_optimal_fees()
