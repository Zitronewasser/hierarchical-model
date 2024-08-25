import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from generateData import num_nodes, degree, grades_arr


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


'''
def adj_to_grade(t_adj_coo):  # 邻接矩阵转换成分数的二维数组，用于画热力图

    grade_2d_arr = np.zeros((degree, num_nodes), dtype=int)
    temp_n = 0
    for coo_row, coo_col in zip(t_adj_coo.row, t_adj_coo.col):
        if temp_n < degree:
            grade_2d_arr[temp_n][coo_row] = grades_arr[coo_col]
            temp_n += 1
        else:
            temp_n = 0
            grade_2d_arr[temp_n][coo_row] = grades_arr[coo_col]
            temp_n += 1
    print("grade_2d_arr:\n", grade_2d_arr)
    return grade_2d_arr
'''


def plot_heatmap(temp_array, t_grades_arr_sum_f):
    # 将数组转置并存储到新的变量中

    temp_array_g = temp_array
    # A_T = np.transpose(temp_array)
    sorted_indices = np.argsort(t_grades_arr_sum_f)[::-1]
    # 使用索引操作进行列变换
    temp_array_g = temp_array_g[:, sorted_indices]
    # 对每一列的元素按照从大到小进行排序
    temp_array_g = np.sort(temp_array_g, axis=0)[::-1]
    # row_numbers = np.arange(num_nodes)[:, np.newaxis]  # 创建一个二维数组，用于存储随机数在'A'中对应的row编号
    # 创建一个二维数组，用于存储随机数在random_integers中对应的编号
    col_numbers = np.arange(degree)[np.newaxis, :]

    # 绘制热力图
    plt.imshow(temp_array_g, cmap='cool', aspect='auto')

    # 设置刻度标签
    # plt.xticks(np.arange(num_nodes), row_numbers[:, 0] + 1)
    plt.xticks(np.arange(0, num_nodes, step=degree * degree))
    plt.yticks(np.arange(degree), col_numbers[0] + 1)

    # 添加颜色标尺
    cbar = plt.colorbar()
    cbar.set_label('Grade')

    # 添加标题和坐标轴标签
    plt.title(f"friends group heatmap (num_node:{num_nodes}) degree:{degree})")
    plt.xlabel('nodes:from highest to lowest')
    plt.ylabel("friends-number")

    # 显示图形
    plt.show()


def show_network(t_graph):
    # 设置边颜色
    edge_color = 'gray'
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(t_graph, with_labels=False, edge_color=edge_color, ax=ax)
    plt.show()


def adj_draw_network(t_coo_matrix, evo_times_t,seed):
    # 创建一个无向图
    adj_graph = nx.Graph()
    # 添加节点
    t_number_nodes = t_coo_matrix.shape[0]
    adj_graph.add_nodes_from(range(t_number_nodes))
    # 添加边
    for i, j in zip(t_coo_matrix.row, t_coo_matrix.col):
        adj_graph.add_edge(i, j)
    # rng = np.random.default_rng(seed)
    # 绘制网络图
    edge_color = 'gray'
    fig, ax = plt.subplots(figsize=(8, 8))
    # nx.draw(adj_graph, with_labels=True, edge_color=edge_color, ax=ax)
    nx.draw(adj_graph, with_labels=True, edge_color=edge_color, ax=ax)
    print("adj_graph:", adj_graph)
    plt.title(f"num_nodes: {num_nodes}, degree: {degree}, evolution times: {evo_times_t} times",
              fontsize=16)  # 添加标题,修改字体大小
    plt.show()


def line_chart(t_grades_sum_f_chart, arr_num, evo_times_t):
    grades_sum_f_chart_tran = np.transpose(t_grades_sum_f_chart)  # 依次分别是9,5,1分的行，转置后的列即表示每一次演化的sum_f

    # 指定要相加并计算平均的行索引
    rows_to_average_9 = np.arange(0, arr_num[0])
    rows_to_average_5 = np.arange(arr_num[0], arr_num[0] + arr_num[1])
    rows_to_average_1 = np.arange(arr_num[0] + arr_num[1], arr_num[0] + arr_num[1] + arr_num[2])

    # 将指定的行相加并计算平均,保留1位小数

    x = np.arange(1, evo_times_t + 1)
    y1 = np.round(np.mean(grades_sum_f_chart_tran[rows_to_average_9], axis=0), decimals=1)
    y2 = np.round(np.mean(grades_sum_f_chart_tran[rows_to_average_5], axis=0), decimals=1)
    y3 = np.round(np.mean(grades_sum_f_chart_tran[rows_to_average_1], axis=0), decimals=1)

    # y1 = grades_sum_f_chart_tran[0]
    # y2 = grades_sum_f_chart_tran[1]
    # y3 = grades_sum_f_chart_tran[2]

    # 绘制折线图
    plt.plot(x, y1, label='grade 9~10')
    plt.plot(x, y2, label='grade 4~5')
    plt.plot(x, y3, label='grade 0~1')

    # 隐藏与 x 轴平行的上边框,隐藏与 y 轴平行的右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks(np.arange(1, evo_times_t+1, step=round(evo_times_t/8)))
    # 自动调整刻度
    plt.xticks()
    # 将刻度值转换为整数
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{int(value)}'))

    # plt.title(f"sum_f Line Chart (num_node:{num_nodes}) degree:{degree})")
    plt.title("The trend of friends in different class")
    plt.xlabel("evolution steps")
    plt.ylabel(" friends average grades")
    plt.legend()
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.25))
    # 显示图形
    plt.show()


def z_optimize(t_grades_sum_f_chart_last_row, arr_num, evo_times_arr_t, z_values_t):
    t_grades_sum_f_chart_last_row_tran = np.transpose(t_grades_sum_f_chart_last_row)  # 依次分别是9,5,1分的行，转置后的列即表示每一个z的sum_f
    # print("t_grades_sum_f_chart_last_row_tran:\n", t_grades_sum_f_chart_last_row_tran)
    # 指定要相加并计算平均的行索引
    rows_to_average_9 = np.arange(0, arr_num[0])
    rows_to_average_5 = np.arange(arr_num[0], arr_num[0] + arr_num[1])
    rows_to_average_1 = np.arange(arr_num[0] + arr_num[1], arr_num[0] + arr_num[1] + arr_num[2])

    # 将指定的行相加并计算平均,保留1位小数

    x = z_values_t  # z的范围就是x轴刻度
    y_p = evo_times_arr_t  # 于y平行的那条轴
    y1_t = np.round(np.mean(t_grades_sum_f_chart_last_row_tran[rows_to_average_9], axis=0) / (9.0 * degree), decimals=1)
    y2_t = np.round(np.mean(t_grades_sum_f_chart_last_row_tran[rows_to_average_5], axis=0) / (5.0 * degree), decimals=1)
    y3_t = np.round(np.mean(t_grades_sum_f_chart_last_row_tran[rows_to_average_1], axis=0) / (1.0 * arr_num[2]),
                    decimals=1)
    y = np.round((y1_t + y2_t) / 2.0, decimals=3)

    # print(f"y1_t: {y1_t}, y2_t:{y2_t}, y3_t:{y3_t}, y:{y}")
    fig, ax1 = plt.subplots()

    # 绘制折线图
    ax1.plot(x, y, color='red', label=' Cluster index')

    # 设置 x 轴刻度
    ax1.set_xticks(x)
    ax1.set_xlabel("z value")
    ax1.set_ylabel("Cluster Index")

    # 创建共享 x 轴的新的 y 轴
    ax2 = ax1.twinx()
    ax2.plot(x, y_p, label='Evolution times')
    ax2.set_ylabel("Evolution Times")

    # 显示图例
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.15))
    ax2.legend(loc='lower right')

    plt.title(f"Z-optimize Line Chart (num_node:{num_nodes} degree:{degree})")
    plt.show()


def plot_box(temp_array_g, t_grades_arr, t2):
    t_indices = []
    for i1 in range(0, 10):
        if np.count_nonzero((i1 < t_grades_arr) & (t_grades_arr <= i1 + 1)) != 0:
            t_indices.append(np.count_nonzero((i1 <= t_grades_arr) & (t_grades_arr < i1 + 1)))

    sorted_indices = np.argsort(t2)  # 升序排序t_grades_arr
    # 使用索引操作进行列变换
    temp_array_g = temp_array_g[:, sorted_indices]

    # 初始化一个空的列表来存储划分后的列的平均值
    result = []

    # 遍历arr中的每个值
    for value in t_indices:
        # 根据arr中的值，选择相应的列并取平均值
        selected_columns = temp_array_g[:, :value]
        column_mean = np.round(np.mean(selected_columns, axis=1), 2)  # 沿列方向计算平均值
        result.append(column_mean)

        # 从temp_array_g中删除已选择的列
        temp_array_g = np.delete(temp_array_g, np.s_[:value], axis=1)

    # 将结果转换为二维数组，每一列是一个箱图的数据集
    result_array = np.array(result).T  # 转置以得到最终的结果,1-10分数的人平均后，每个人朋友的分数，一共10列，行数和temp_array_g相同
    # print(result_array)

    # 使用boxplot函数绘制多个箱图
    plt.boxplot(result_array)

    # 自定义y轴刻度为1到10的正整数
    plt.yticks(np.arange(1, 11, 1))

    # 添加标题和标签
    # plt.title('Box Plots')
    plt.xlabel('different level of nodes')
    plt.ylabel('Grade')

    # 显示多个箱图
    plt.show()


def calculate_percentage(t_coo_matrix, t_grades_arr):
    t_indices = []
    for i1 in range(1, 11):
        if np.count_nonzero(t_grades_arr == i1) != 0:
            t_indices.append(np.count_nonzero(t_grades_arr == i1))

    sorted_indices = np.argsort(t_grades_arr)  # 升序排序t_grades_arr
    # 使用索引操作进行列变换
    t_coo_matrix = t_coo_matrix[:, sorted_indices]

    # 要统计的特定元素值
    target_value = 2
    t_coo_matrix = t_coo_matrix.tocoo()
    # 统计每一行中特定元素值的个数
    row_count = []
    for row_idx in range(t_coo_matrix.shape[0]):
        row_indices = np.where(t_coo_matrix.row == row_idx)[0]
        count = np.sum(t_coo_matrix.data[row_indices] == target_value)
        count = float(count)
        count = round(count / degree, 2)
        row_count.append(count)

    result = []
    for value in t_indices:
        # 根据arr中的值，选择相应的列并取平均值
        selected_columns = row_count[:value]
        column_mean = np.round(np.mean(selected_columns), 2)  # 沿列方向计算平均值
        result.append(column_mean)
        # 从temp_array_g中删除已选择的列
        row_count = np.delete(row_count, np.s_[:value])

    # 打印每一行中特定元素值的个数
    # print("每一行中特定元素值的个数:")
    # for row_idx, count in enumerate(row_count):
    #     print(f"第 {row_idx} 行: {count} 个")

    # Create a table image
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create a table object
    table_data = []
    for i, count in enumerate(result):
        table_data.append([f'Grade {i + 1}', count])

    # Table column labels and data
    col_labels = ['Grade', 'old friends percentage']

    # Create the table
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Adjust the table layout
    table.scale(1, 1.5)

    # Save the table image as a PNG file
    plt.savefig('table_image.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

    # Display the table image (optional)
    plt.show()


def sums_histogram(t_arr1):
    # 绘制直方图
    degree_f = degree+1
    # custom_bins = np.arange(degree_f * 1, degree_f * 10, step=2)
    # custom_bins = [50,20,30, 40, 45, 50,60,70,80]
    max_arr_value = max(t_arr1)  # 获取最大值
    min_arr_value = min(t_arr1)  # 获取最小值
    custom_bins = np.linspace(10, 1620, 8)



    plt.hist(t_arr1, bins=custom_bins, edgecolor='black', color='#add8a4')

    # 设置标题和轴标签
    plt.title(f"Histogram of sum of friends' scores (num_node:{num_nodes} degree:{degree_f})")
    plt.xlabel("Sum of friends' scores")
    plt.ylabel("Frequency")
    # plt.xticks(np.arange(degree_f * 1, degree_f * 10, step=degree_f))
    plt.xticks(custom_bins)
    plt.show()


def plot_points(t_evo_times, t_num_connected):
    # 生成曲线上的点
    x = np.linspace(0, t_evo_times, t_evo_times+1, dtype=int)  # 生成 x 值
    y = t_num_connected  # 生成 y 值
    y = [yt/(num_nodes/(degree+1)) for yt in y]
    # 绘制点图，点形状为蓝色空心圆圈
    plt.plot(x, y, marker='o', linestyle='None', markersize=8, markerfacecolor='none', markeredgecolor='blue')

    # 设置坐标轴标签
    plt.xlabel('evolving steps')
    plt.ylabel('Degree of division')
    plt.xticks(np.arange(0,t_evo_times, step=1))
    plt.yticks(np.linspace(0, 1, 6))
    evo_degree = 100*t_num_connected[-1]/(num_nodes/(degree+1))

    # 设置图标题
    # plt.title(f"num_node:{num_nodes} degree:{degree},evo_percentage:{round(evo_degree,0)}%")
    plt.title("The trend of connected clusters ")
    # 显示图形
    plt.show()


def cc_and_variance(pre_cc_node_nine, pre_cc_node_five, pre_cc_node_one, pre_cc_network,t_evo_times):
    # # 计算每组数据的方差
    # variances = [np.var(d) for d in data]

    x = np.linspace(1, t_evo_times, t_evo_times, dtype=int)

    # 点用深绿色的三角形
    plt.plot(x, pre_cc_node_nine, label='score_9~10', marker='^', color='#28817f', linestyle='None', markersize=5)
    plt.plot(x, pre_cc_node_nine, color='#74adac', linestyle='-')

    plt.plot(x, pre_cc_node_five, label='score_4~5', marker='^', color='#3f85ff', linestyle='None', markersize=5)
    plt.plot(x, pre_cc_node_five, color='#3f85ff', linestyle='-')

    plt.plot(x, pre_cc_node_one, label='score_0~1', marker='^', color='orange', linestyle='None', markersize=5)
    plt.plot(x, pre_cc_node_one, color='orange', linestyle='-')

    plt.plot(x, pre_cc_network, label='Network', marker='s', color='purple', linestyle='None', markersize=5)
    plt.plot(x, pre_cc_network, color='purple', linestyle='-')

    plt.xlabel('evolving steps')
    plt.ylabel('cluster coefficient ')
    # plt.xticks(np.arange(1, t_evo_times, step=t_evo_times // 6))
    plt.xticks()
    plt.title(f"Trend of cluster coefficient of network and different class")
    plt.legend()

    plt.show()


def plot_variance(vector_list_2d,t_evo_times):
    # pre_vector_nine = np.zeros(num_nodes, dtype=int)
    # score_nine_indices = np.where(grades_arr == 9)
    # first_score_nine_indices = score_nine_indices[0][0]
    # for col1 in t_coo_matrix.col[t_coo_matrix.row == first_score_nine_indices]:  # 仿照col_index_to_grade函数
    #     pre_vector_nine[col1] = grades_arr[col1]  # 此处为读取score_standard里i1的朋友的标准分数
    # vector_score_friend_nine = pre_vector_nine[pre_vector_nine != 0]  # 使用布尔索引提取非零元素

    # 计算每组数据的方差

    # variances = [np.var(d) for d in vector_list_2d]
    variances = [np.mean(d) for d in vector_list_2d]
    variances = [round(var1, 3) for var1 in variances]
    x = np.linspace(0, t_evo_times, t_evo_times+1, dtype=int)

    plt.plot(x, variances, label='Mean', marker='s', color='#28817f', linestyle='None', markersize=5)
    plt.plot(x, variances, color='#74adac', linestyle='-')

    plt.xlabel('evolution time')
    plt.ylabel('Mean')
    plt.xticks(np.arange(1, t_evo_times, step=t_evo_times // 4))
    plt.title(f"Mean friends score of score 9 node (num_node:{num_nodes} degree:{degree})")
    plt.legend()
    plt.show()



def calculate_whiskers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    return lower_whisker, upper_whisker


def plot_mean_and_whiskers(vector_list_2d, t_evo_times):
    # 计算每组数据的均值
    means = [np.mean(d) for d in vector_list_2d]
    means = [round(mean, 3) for mean in means]

    # 计算每组数据的下须和上须
    lower_whiskers = []
    upper_whiskers = []
    for d in vector_list_2d:
        lower_whisker, upper_whisker = calculate_whiskers(d)
        lower_whiskers.append(lower_whisker)
        upper_whiskers.append(upper_whisker)

    # 准备绘制数据
    x = np.linspace(0, t_evo_times, t_evo_times + 1, dtype=int)

    # 绘制均值曲线
    plt.plot(x, means, label='Mean', marker='s', color='#28817f', linestyle='None', markersize=5)
    plt.plot(x, means, color='#74adac', linestyle='-')

    # 绘制上须和下须曲线
    plt.plot(x, upper_whiskers, label='Upper Whisker', marker='^', color='red', linestyle='None', markersize=5)
    plt.plot(x, upper_whiskers, color='pink', linestyle='-')

    plt.plot(x, lower_whiskers, label='Lower Whisker', marker='v', color='blue', linestyle='None', markersize=5)
    plt.plot(x, lower_whiskers, color='lightblue', linestyle='-')

    # 添加图例和标签
    plt.xlabel('Evolution Time')
    plt.ylabel('Value')
    plt.xticks(np.arange(1, t_evo_times, step=t_evo_times // 8))
    plt.title(f"Mean and Whiskers of Score 9 Node (num_node:{num_nodes} degree:{degree})")
    plt.legend()
    plt.show()


def calcu_cos_sim(t_coo_matrix, t_coo_matrix_ini):

    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity

    def coo_to_vector_t(t1_coo_matrix, score_node_index):  # 这个函数会返回向量(列表)
        pre_vector_score_friend = np.zeros(num_nodes, dtype=int)
        # score_nine_indices = np.where(grades_arr == score_node)
        # first_score_nine_indices = score_nine_indices[0][0]

        for col1 in t1_coo_matrix.col[t1_coo_matrix.row == score_node_index]:  # 仿照col_index_to_grade函数
            pre_vector_score_friend[col1] = grades_arr[col1]  # 此处为读取score_standard里i1的朋友的标准分数

        vector_score_friend = pre_vector_score_friend[pre_vector_score_friend != 0]  # 使用布尔索引提取非零元素
        vector_score_friend = vector_score_friend.tolist()
        return vector_score_friend


    x_values = [1, 3, 5, 7, 9]

    sim_2d_list = []
    for j in x_values:
        score_nine_indices = np.where(grades_arr == j)
        sim_specific_node = []
        for i in range(len(score_nine_indices[0])):  # 用到所有的9

            nine_index = score_nine_indices[0][i]
            v_score_friend = coo_to_vector_t(t_coo_matrix, nine_index)
            v_score_friend_ini = coo_to_vector_t(t_coo_matrix_ini, nine_index)
            similarity_temp = cosine_similarity(v_score_friend,v_score_friend_ini)
            similarity_temp = round(similarity_temp, 3)
            sim_specific_node.append(similarity_temp)

        sim_2d_list.append(sim_specific_node)


    # y_values = np.array(sim_2d_list)
    y_values = sim_2d_list
    x_coords = []  # 动态处理
    y_coords = []
    for x, y_list in zip(x_values, y_values):
        x_coords.extend([x] * len(y_list))
        y_coords.extend(y_list)

    # 绘制点图
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='^', linestyle='None', markersize=8, markerfacecolor='none',
             markeredgecolor='#23966f')

    plt.xlabel('Node score')
    plt.ylabel('Cosine similarities')
    plt.xticks(x_values)
    plt.title(f"Similarities of different score of nodes (num_node:{num_nodes} degree:{degree})")

    plt.show()


def plot_diff(diff_list,evo_times):

    # print("len",len(diff_list))

    x = np.linspace(1, evo_times, evo_times, dtype=int)

    sum_diff_list =[]
    for j in range(len(diff_list)):
        sum_diff_list.append(sum(diff_list[j]))

    x_1 = np.linspace(1, len(sum_diff_list), len(sum_diff_list), dtype=int)
    # Plot multiple lines with triangle markers
    for i, row in enumerate(diff_list):
        plt.plot(x, row, marker='^', label=f'Line {i + 1}')

    # plt.plot(x_1, sum_diff_list, marker='^', label='score 6')


    plt.xlabel('evolution tiome')
    plt.ylabel('Difference Value')
    plt.xticks(np.arange(1, evo_times+2, step=evo_times // 8))
    plt.title(f"difference of friends of score 9 (num_node:{num_nodes} degree:{degree})")

    # plt.xlabel('index plus 1 of each score 9 ')
    # plt.ylabel('Sum Difference Value')
    # plt.title(f"Sum difference  of each score 9 (num_node:{num_nodes} degree:{degree})")

    plt.legend()

    # Display the chart
    plt.show()



