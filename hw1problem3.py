import random
from random import sample
import numpy as np
#
# def cointoss():
#     # 1 for head and 0 for tail
#     result = np.random.randint(2, size = 3)
#     return result

def pred(n, y):
    y_test = []
    for i in range(n):
        tosses = np.random.binomial(3, 0.5)
        if tosses > 0:
            y_test.append(y[i])
        else:
            y_test.append(1-y[i])
    return y_test


def generate_sample(n):
    pop = []
    for i in range(n):
        random_select = np.random.choice(np.arange(2), p=[0.9, 0.1])
        pop.append(random_select)
    return pop

if __name__ == '__main__':
    num_trials = 10
    mean=[]
    std_dev=[]
    for n in [100, 200, 400, 800, 1600]:
        test_propotion = np.zeros(num_trials)
        for t in range(num_trials):
            y = generate_sample(n)
            predict = pred(n, y)
            num_pos = sum(predict)
            theta = 4/3 * ((num_pos-1/8 * n) / n)
            test_propotion[t] = theta
        mean.append(round(np.mean(test_propotion),3))
        std_dev.append(round(np.std(test_propotion),3))
        print("n: ", n)
        print("mean: ", mean[-1])
        print("std dev: ", std_dev[-1])
    print()
    print("n: ",[100, 200, 400, 800, 1600])
    print("mean: ", mean)
    print("std dev: ", std_dev)




