# %%
import numpy
import os
import pickle
from tqdm import tqdm

trans_P = None
n_car = 20

def _poisson(
    location:int,
    option:str,
):    
    _lambda = {
        'rental': {
            1: 3,
            2: 4,
        },
        'return': {
            1: 3,
            2: 2
        }
    }
    _lambda = _lambda[option][location]

    result = []
    for i in range(21):
        poisson = (numpy.power(_lambda, i) / numpy.math.factorial(i)) * numpy.exp(-_lambda)
        result.append(poisson)
        
    return result

# %%
def p(  # This is the p in Policy Itertaion at 80 page. p(s', r | s, pi(s))
    joint_prob,
    credit=10,
    cost_car_move=2,
    cost_parking=4,
    conditional_prob:dict=None
):
    """_summary_

    Args:
        joint_prob: All posibility about (rental_1, return_1, rental_2, return_2)
        credit (int, optional): _description_. Defaults to 10.
        cost_car_move (int, optional): _description_. Defaults to 2.
        cost_parking (int, optional): _description_. Defaults to 4.

    Returns:
        Given current state and action(policy),
        the next all pair of (state, reward) and corresponding possibility.
        This is p in Section 4.3 Policy Iteration at page 80.
        ->
        Dict:
            key: ((state_1, state_2)) = (state)
            value: Dict:
                key: action
                value: Dict:
                    key: ((next_state_1, next_state_2), reward) = (state', reward)
                    value: probability.
    """
    for state_1 in tqdm(range(n_car + 1)):    # Current remaining # cars at 1
        for state_2 in range(n_car + 1):
            prob_given_action_state = {}
            for action in range(-5, 6): # Move cars from 1 to 2 location. ex)5: 5 cars are moved from 1 to 2, -5: 5 cars are moved from 2 to 1.
                temp = {}
                if  (0 <= state_1 - action) and (0 <= state_2 + action) and (action + state_2 <= n_car) and (-action + state_1 <= n_car):
                    for rental_1 in range(n_car + 1):   # Total rental car at the end of the day at 1.
                        for rental_2 in range(n_car + 1):
                            for return_1 in range(n_car + 1):
                                for return_2 in range(n_car + 1):
                                    sell_car_1 = min(rental_1, state_1 - action)
                                    sell_car_2 = min(rental_2, state_2 + action)
                                    changed_state_1 = min(  # Changed # cars over night at 1.
                                        20, state_1 - sell_car_1 + return_1 - action   # state -> sell -> return -> action -> changed_state
                                    )
                                    changed_state_2 = min(  # Changed # cars over night at 2.
                                        20, state_2 - sell_car_2 + return_2 + action   # state -> sell -> return -> action -> changed_state
                                    )
                                    
                                    benefit = (
                                        credit * sell_car_1 + 
                                        credit * sell_car_2 
                                    )
                                    cost = (
                                        cost_car_move * abs(action if action <= 0 else (action - 1)) +
                                        cost_parking if changed_state_1 >= 10 else 0 +
                                        cost_parking if changed_state_2 >= 10 else 0
                                    )
                                    reward = benefit - cost
                                    new_state = (changed_state_1, changed_state_2)
                                    
                                    if (new_state, reward) not in temp.keys():
                                        temp[(new_state, reward)] = joint_prob[(rental_1, return_1, rental_2, return_2)]
                                    else:
                                        temp[(new_state, reward)] += joint_prob[(rental_1, return_1, rental_2, return_2)]
                    prob_given_action_state[action] = temp
            current_state = (state_1, state_2)
            conditional_prob[(current_state)] = prob_given_action_state.copy()
    return conditional_prob

# %%
def init():
    print("Initialization")
    loc_1_rental_prob = _poisson(1, 'rental')
    loc_2_rental_prob = _poisson(2, 'rental')
    loc_1_return_prob = _poisson(1, 'return')
    loc_2_return_prob = _poisson(2, 'return')
        
    joint_prob = {}
    
    print("Calculate all joint probability")
    for i in range(21):
        for j in range(21):
            for k in range(21):
                for l in range(21):
                    joint_prob[(i, j, k, l)] = loc_1_rental_prob[i] * loc_1_return_prob[j] * loc_2_rental_prob[k] * loc_2_return_prob[l]
    
    print("Calculate conditional probability")
    conditional_prob = {}
    p(
        joint_prob=joint_prob,
        conditional_prob=conditional_prob
    )
    return conditional_prob
    
def policy_evaluation(
    value_func,
    threshold,
    p,
    policy:dict,
    discount_factor:float=0.9
):
    print("Policy evaluation")
    while True: # Loop
        delta = 0   # delta <- 0
        for state in p.keys():  # Loop for each s in S.
            action = policy[state]
            all_next_state_reward_prob = p[state][action] # The all probability of (next_state, reward)
            old_value =  value_func[state]
            value_func[state] = 0
            for (all_state, reward), prob in all_next_state_reward_prob.items(): # V(s) <- Summation for p(s', r|s, policy(s))[r + gamma V(s')]
                value_func[state] += prob * (reward + discount_factor * value_func[all_state]) 
            delta = max(delta, abs(old_value - value_func[state]))  # delta <- max(delta, |v - V(s)|)
            print(f"\rdelta - threshold: {delta - threshold}" + 200*" ", flush=True, end='')
        if delta < threshold:
            print()
            return value_func

def policy_improvement(
    value_func,
    p,
    policy:dict={},
    discount_factor:float=0.9
):
    print("Policy improvement")
    policy_stable = True    # policy-stable <- True
    while policy_stable:
        for state, old_action in policy.items():
            state_1, state_2 = state
            temp_value_list = list()
            for possible_action in range(-5, 6):  # for all actions (in argmax_a summation ~~~)
                temp_value = 0
                if  (0 <= state_1 - possible_action) and (0 <= state_2 + possible_action) and (possible_action + state_2 <= n_car) and (-possible_action + state_1 <= n_car):
                    for (all_state, reward), prob in p[state][possible_action].items():    # for all possible states. (in argmax_a summation ~~~)
                        temp_value += prob * (reward + discount_factor * value_func[all_state])
                else:   # 
                    temp_value = -99999

                temp_value_list.append(temp_value if type(temp_value) == int else temp_value[0])
            policy[state] = numpy.argmax(temp_value_list) - 5
            
            if policy[state] != old_action:
                policy_stable = False
    return policy
            

# %%
if __name__ == "__main__":
    if not os.path.exists('./6.pkl'):
        p = init()
        with open('./6.pkl', 'wb') as f:
            pickle.dump(p, f)
    else:
        print("Loading saved data")
        with open('./6.pkl', 'rb') as f:
            p = pickle.load(f)

    value_func = {}
    policy = {}
    for state_1 in range(21):
        for state_2 in range(21):
            state = (state_1, state_2)
            value_func[state] = numpy.random.standard_normal(
                size=1
            )
            policy[state] = 0

    value_func = policy_evaluation(
        value_func=value_func,
        policy=policy,
        threshold=5e-3,
        p=p
    )
    # %%
    policy = policy_improvement(
        value_func=value_func,
        policy=policy,
        p=p
    )
    # %%    
    import os
    os.system("pip install seaborn")
    import seaborn as sns
    result = (value_func, policy)

    policy_heatmap = numpy.zeros((21, 21))

    for state, action in policy.items():
        policy_heatmap[state] = action
    sns.heatmap(policy_heatmap)    
        
        

print("Exit")