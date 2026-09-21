import math 

import random
rng = random.Random(67)

"""
48 scenari da 10 min
3 topologie
"""

def sample_attack_rate(npkts, flow_table_size, min_anp={"aggressive": 10,"moderate": 7,"stealth": 2}):
    stealth_profiles = {
        "aggressive": (10, 60),
        "moderate":   (120, 300),
        "stealth":    (350, 550),
    }
    profile = rng.choice(list(stealth_profiles.keys()))
    t_min, t_max = stealth_profiles[profile]
    fill_time = rng.uniform(t_min, t_max)
    anp = max(min_anp[profile], math.ceil(flow_table_size / fill_time))
    #print(math.ceil(flow_table_size / fill_time))
    return anp, profile, fill_time

n_host = 15
full_size = int(math.factorial(n_host) / math.factorial(n_host-2)) + 1
half_size = int(full_size / 2)
print(half_size)

for i in range(49):
    # hard_timeout: 5 < hard <= 30
    hard_timeout = rng.randint(6, 15)
    # idle_timeout: idle <= hard
    idle_timeout = rng.randint(2, hard_timeout)
    # flow_table_size
    flow_table_size = rng.randint(half_size, 2*full_size)
    # att_npkts: att_npkts >= flow_table_size
    npkts = flow_table_size + rng.randint(0, half_size)

    anp, profile, fill_time = sample_attack_rate(npkts, flow_table_size)

    print(f"> Sim {i}")
    print(f"hard timeout: {hard_timeout}, idle timeout: {idle_timeout}, flow tale size: {flow_table_size}, npkts: {npkts}")
    print(f"anp: {anp}, profile: {profile}, fill time: {fill_time}\n")
