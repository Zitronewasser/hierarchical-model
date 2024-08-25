import numpy as np

# 生成的总人数
SumP = 9
PIHouse = 3  # People in every House
# 设置种子（seed）
seed = 42


def generate_data(sum_p, pi_house):
    global NumHouse, random_integers, reshaped_array, A
    NumHouse = int(sum_p / pi_house)
    # 创建一个随机数生成器对象
    rng = np.random.RandomState(seed)

    # 设置均值和标准差
    mu = 5  # 均值
    sigma = 2  # 标准差

    # 生成服从正态分布的随机数
    random_numbers = rng.normal(mu, sigma, sum_p)

    # 将随机数限制在1到10之间的范围
    random_numbers = np.clip(random_numbers, 1, 10)

    # 将浮点数转换为整数
    random_integers = np.round(random_numbers).astype(int)

    # 将一维数组重新组织成二维数组，每行包含 pi_house 个数
    reshaped_array = random_integers.reshape(NumHouse, pi_house)

    # 将每行转换为列表，并存储在一个列表中
    # A = [list(row) for row in reshaped_array]
    A = reshaped_array

    return random_integers, sum_p, pi_house, NumHouse, reshaped_array, A


random_integers, SumP, PIHouse, NumHouse, reshaped_array, A = generate_data(SumP, PIHouse)

# # 将生成的随机数打印出来
# print(random_integers)
# # 打印结果
# for i, arr in enumerate(A):
