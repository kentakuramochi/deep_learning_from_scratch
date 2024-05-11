""" Iterative policy evaluation by dynamic programming (DP).

An example of MDP on the gridworld:

     L1    L2       State: L1, L2
   +-----+-----+
-1 | 0   | +1  | -1 Reward: -1, 0, 1
   +-----+-----+
     <-L   R->      Action: Left, Right

* The agent moves randomly:
    pi(a|s) = 0.5.

* The transision of the state is definitive:
    p(L1|L1, Left) = 1
    p(L2|L1, Right) = 1
    p(L1|L2, Left) = 1
    p(L2|L2, Ritht) = 1
    p(s'|s, a) = 0 otherwise
"""

# State value function and those initial values
V = {"L1": 0.0, "L2": 0.0}

# State transition probability
p = 0.5
# Discount rate
gamma = 0.9
# Rewards by (s, a, s')
r = {"L1_Left_L1": -1, "L1_Right_L2": 1, "L2_Left_L1": 0, "L2_Right_L2": -1}

# Iterate and update the esimated state value functions
cnt = 0
while True:
    t = p * (r["L1_Left_L1"] + gamma * V["L1"]) + p * (
        r["L1_Right_L2"] + gamma * V["L2"]
    )
    delta = abs(t - V["L1"])  # Max diff. of updates
    V["L1"] = t

    # Updated V["L1"] is used immediately
    t = p * (r["L2_Left_L1"] + gamma * V["L1"]) + p * (
        r["L2_Right_L2"] + gamma * V["L2"]
    )
    delta = max(delta, abs(t - V["L2"]))
    V["L2"] = t

    cnt += 1
    # Finish update
    # True values: v(L1)=-2.25, v(L2)=-2.75
    if delta < 0.0001:
        print(V)
        print(cnt)
        break
