import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 生成随机数
num_nodes = 9*9*10  # 节点数量
degree = 9  # 每个节点的度数
seed = 3
mu = 5  # 均值
sigma = 2  # 标准差

random.seed(seed)
rng = np.random.RandomState(seed)

random_numbers = rng.normal(mu, sigma, num_nodes)
random_numbers = np.clip(random_numbers, 1, 10)

unique_random_numbers = list(set(round(num, 3) for num in random_numbers))

while len(unique_random_numbers) < num_nodes:
    extra_numbers = rng.normal(mu, sigma, num_nodes - len(unique_random_numbers))
    extra_numbers = np.clip(extra_numbers, 1, 10)
    unique_random_numbers.extend(round(num, 3) for num in extra_numbers)
    unique_random_numbers = list(set(unique_random_numbers))

grades_arr = np.array(unique_random_numbers[:num_nodes])

# 绘制Q-Q图
plt.figure(figsize=(10, 6))
stats.probplot(grades_arr, dist="norm", plot=plt)

# 设置标题
plt.title("Q-Q Plot of Random Numbers")

# 显示图形
plt.show()
