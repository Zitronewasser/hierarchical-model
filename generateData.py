import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix



def plot_histogram(t_random_integers):
    # 绘制直方图
    plt.hist(t_random_integers, bins=10, range=(1, 10), edgecolor='black', color='lightblue')

    # 设置标题和轴标签
    plt.title(f"Histogram of Random Integers (num_node:{num_nodes}) degree:{degree})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 设置横坐标刻度位置和标签
    plt.xticks(range(1, 11))

    # 显示图形
    plt.show()


num_nodes = 8*8*9 # 节点数量 6*6*7 8*8*9
degree = 8 # 每个节点的度数,即每个人拥有的朋友数量,num_nodes需要整除（degree+1），每个小团体是（degree+1）人

# 设置网络参数
seed = 3
# min_seed = 0
# max_seed = 2**32 - 1
# seed = random.randint(min_seed, max_seed)

random.seed(seed)
print("seed:", seed)
rng = np.random.RandomState(seed)

# 设置均值和标准差
mu = 5  # 均值
sigma = 2  # 标准差


# random_numbers = rng.normal(mu, sigma, num_nodes)
# random_numbers = np.clip(random_numbers, 1, 10)
# grades_arr = np.round(random_numbers).astype(int)

# 生成100个服从正态分布的随机数
random_numbers = rng.normal(mu, sigma, num_nodes)
# 将随机数限制在1到10之间的范围
random_numbers = np.clip(random_numbers, 1e-1, 10)

# 保留三位小数并去除重复的数值
unique_random_numbers = list(set(round(num, 3) for num in random_numbers))

# 如果去重后数量不足100个，则继续生成并补充
while len(unique_random_numbers) < num_nodes:
    extra_numbers = rng.normal(mu, sigma, num_nodes - len(unique_random_numbers))
    extra_numbers = np.clip(extra_numbers, 1, 10)
    unique_random_numbers.extend(round(num, 3) for num in extra_numbers)
    unique_random_numbers = list(set(unique_random_numbers))

# 截取前100个数
grades_arr =  np.array(unique_random_numbers[:num_nodes])
# print(sorted(grades_arr))
# plot_histogram(grades_arr)
# 生成Homogeneous Random Network随机网络
homogeneous_graph = nx.random_regular_graph(degree, num_nodes,seed=seed)

adj_matrix = nx.to_numpy_array(homogeneous_graph).astype(int)  # 生成稠密邻接矩阵,并不让人成int类型
adj_matrix_coo_ini = coo_matrix(nx.to_scipy_sparse_array(homogeneous_graph)).astype(int)  # 将稠密邻接矩阵转换为 COO 格式的稀疏矩阵
adj_matrix_csr = adj_matrix_coo_ini.tocsr()  # 将 COO 格式的稀疏矩阵转换为 CSR 格式的稀疏矩阵


def compute_sec_adj(t_adj_matrix_csr):
    secondary_adj_csr = t_adj_matrix_csr.dot(t_adj_matrix_csr)  # 矩阵乘法
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)
    secondary_adj_csr = secondary_adj_csr - t_adj_matrix_csr  # 减去邻接矩阵
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)

    I_arr = csr_matrix(np.eye(num_nodes, dtype=int))
    secondary_adj_csr = secondary_adj_csr - I_arr
    # secondary_adj_csr = secondary_adj_csr.tocoo()
    return secondary_adj_csr


def compute_third_adj(t_adj_matrix_csr):
    # 计算二级邻居
    secondary_adj_csr = t_adj_matrix_csr.dot(t_adj_matrix_csr)
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)
    secondary_adj_csr = secondary_adj_csr - t_adj_matrix_csr
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)
    I_arr = csr_matrix(np.eye(num_nodes, dtype=int))


    # 计算三级邻居
    third_adj_csr = secondary_adj_csr.dot(t_adj_matrix_csr)
    third_adj_csr.data = np.where(third_adj_csr.data > 0, 1, 0)
    third_adj_csr = third_adj_csr -t_adj_matrix_csr
    third_adj_csr.data = np.where(third_adj_csr.data > 0, 1, 0)

    third_adj_csr = third_adj_csr - I_arr
    third_adj_csr.data = np.where(third_adj_csr.data > 0, 1, 0)

    return third_adj_csr


choice_adj_csr = csr_matrix(adj_matrix_coo_ini) + compute_sec_adj(csr_matrix(adj_matrix_coo_ini))# 邻接矩阵+次级邻居 + compute_third_adj(adj_matrix_coo_ini)
choice_adj_coo = choice_adj_csr.tocoo()



# score_nine_indices_t = np.where(grades_arr == 9)
# nine_index_t = score_nine_indices_t[0][0]

# 预设山村条件
# for neighbor in choice_adj_coo.col[choice_adj_coo.row == nine_index_t]:
#     grades_arr[neighbor] = random.randint(1, 4)


# score_nine_indices_t = np.where(grades_arr == 9)
# nine_index_t = score_nine_indices_t[0][0]
# pre_vector_score_friend_t = np.zeros(num_nodes, dtype=int)
# for col1 in choice_adj_coo.col[choice_adj_coo.row == nine_index_t]:  # 仿照col_index_to_grade函数
#     pre_vector_score_friend_t[col1] = grades_arr[col1]  # 此处为读取score_standard里i1的朋友的标准分数
#
# vector_score_friend = pre_vector_score_friend_t[pre_vector_score_friend_t != 0]  # 使用布尔索引提取非零元素
# vector_score_friend = vector_score_friend.tolist()
# print("vector_score_friend:", vector_score_friend)