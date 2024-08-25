import numpy as np
from statistic import show_network, plot_histogram, plot_heatmap, adj_draw_network, line_chart
from scipy.sparse import coo_matrix, csr_matrix
from generateData import grades_arr, homogeneous_graph, adj_matrix_coo_ini, num_nodes, degree
import time
# 记录开始时间
start_time = time.time()

print("num_nodes：", num_nodes, "degree：", degree)
print("homogeneous_graph:", homogeneous_graph)
# print("grades_arr:\n", grades_arr)

adj_matrix_coo = adj_matrix_coo_ini.copy()

# np.set_printoptions(threshold=np.inf)  # 设置打印选项，使其显示完整的数组
# np.set_printoptions(linewidth=np.inf)  # 设置打印选项，禁止自动换行
# print("Dense adj_matrix_coo:\n")
# print(adj_matrix_coo_ini.toarray())




sorted_indices = np.argsort(grades_arr)[::-1]  # 对分数数组进行排序，默认是从小到大排序，故返回反转后的索引数组，即返回的是从大到小的
# print("sorted_indices:", sorted_indices)
# print("最大分数的索引值:", sorted_indices[0])
# print("最大分数:", grades_arr[sorted_indices[0]])


def modify_adj_m_to_zero(t_adj_matrix_coo, row_idx, col_idx):
    # 判断指定位置的元素是否在 COO 格式稀疏矩阵的存储中
    mask = (t_adj_matrix_coo.row == row_idx) & (t_adj_matrix_coo.col == col_idx)
    if mask.any():
        # 获取指定位置元素的索引
        index = np.where(mask)[0][0]

        # 删除指定位置的元素
        t_adj_matrix_coo.row = np.delete(t_adj_matrix_coo.row, index)
        t_adj_matrix_coo.col = np.delete(t_adj_matrix_coo.col, index)
        t_adj_matrix_coo.data = np.delete(t_adj_matrix_coo.data, index)
    else:
        print("warning!!  This value is not 1")


def modify_adj_m_to_one(t_adj_matrix_coo, row_idx, col_idx):
    # 判断指定位置的元素是否在 COO 格式稀疏矩阵的存储中
    if ((t_adj_matrix_coo.row == row_idx) & (t_adj_matrix_coo.col == col_idx)).any():
        print("D!!, it's already 1")
    else:
        # 在 COO 格式稀疏矩阵的存储中插入新元素的信息
        insert_idx = np.searchsorted(t_adj_matrix_coo.row, row_idx)
        t_adj_matrix_coo.row = np.insert(t_adj_matrix_coo.row, insert_idx, row_idx)
        t_adj_matrix_coo.col = np.insert(t_adj_matrix_coo.col, insert_idx, col_idx)
        t_adj_matrix_coo.data = np.insert(t_adj_matrix_coo.data, insert_idx, 1)


# 计算次级邻居的矩阵
def compute_sec_adj(t_adj_matrix_csr):
    secondary_adj_csr = t_adj_matrix_csr.dot(t_adj_matrix_csr)  # 矩阵乘法
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)
    secondary_adj_csr = secondary_adj_csr - t_adj_matrix_csr  # 减去邻接矩阵
    secondary_adj_csr.data = np.where(secondary_adj_csr.data > 0, 1, 0)
    I_arr = csr_matrix(np.eye(num_nodes, dtype=int))
    secondary_adj_csr = secondary_adj_csr - I_arr
    # secondary_adj_csr = secondary_adj_csr.tocoo()
    return secondary_adj_csr


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
    return grade_2d_arr


# print("secondary_adj_csr:")
# print(sec_adj_csr.toarray())  # 转换为 NumPy 数组并打印
# modified_adj_matrix_coo = modify_adj_matrix(adj_matrix_coo, 6, 6)


# print("sorted_indices:", sorted_indices)
# print("grades_arr：", grades_arr)
lack_of_f_matrix = np.zeros(num_nodes, dtype=int)  # 表示一个人现在缺少了几个朋友，元素为负值


def col_index_to_grade(t_coo_matrix, t_row_number):  # 需要传入一个coo矩阵，和一个行标，此函数会把那行为1的元素映射成他们的分数
    t_arr1d_for_sort_g = np.zeros(num_nodes, dtype=int)
    for col in t_coo_matrix.col[t_coo_matrix.row == t_row_number]:
        t_arr1d_for_sort_g[col] = grades_arr[col]
    return t_arr1d_for_sort_g


def drop_the_lowest_g(t_coo_matrix, t_row_number):  # 这个函数把接收到的这个行标的朋友里分数最低的那个人drop掉
    each_f_grade_list = col_index_to_grade(t_coo_matrix, t_row_number)
    each_f_nonzero_count = len(t_coo_matrix.data[t_coo_matrix.row == t_row_number])  # 这是t_row_number现有的朋友的数量
    drop_col_index = np.argsort(each_f_grade_list)[-each_f_nonzero_count:]  # 这是一个分数从低高到排序后返回的列索引，只保留倒数each_f_nonzero_count个
    f_be_dropped_col_num = drop_col_index[0]  # 分数最低的那个倒霉蛋
    modify_adj_m_to_zero(t_coo_matrix, t_row_number, f_be_dropped_col_num)
    modify_adj_m_to_zero(t_coo_matrix, f_be_dropped_col_num, t_row_number)
    lack_of_f_matrix[f_be_dropped_col_num] -= 1  # 因为这个倒霉蛋（f_be_dropped_col_num），他是被迫失去了一个朋友，所以给他-1


def drag_count_fun(t_lack_of_f_matrix):  # 返回当下这些drag的数量
    t_count_drag = 0
    for t2_col in range(num_nodes):
        if t_lack_of_f_matrix[t2_col] != 0:
            t_count_drag += 1
    return t_count_drag


def get_coo_value(t_adj_matrix_coo, row_idx, col_idx):  # 返回coo格式存储的矩阵特定位置的元素值
    bool_arr = (t_adj_matrix_coo.row == row_idx) & (t_adj_matrix_coo.col == col_idx)
    if np.any(bool_arr):
        # 获取指定位置元素的索引
        index = np.where(bool_arr)[0][0]
        # print("index:", index)
        element_value = t_adj_matrix_coo.data[index]
    else:
        element_value = 0
    return element_value


# show_network(homogeneous_graph)
# plot_heatmap(adj_to_grade(adj_matrix_coo))

'''
# 粗演化
for high_g_node in sorted_indices:
    if lack_of_f_matrix[high_g_node] < 0:
        print(f"number {high_g_node} is still lack of {np.abs(lack_of_f_matrix[high_g_node])} friends")
        print("The process has to end")
        break
    else:

        choice_adj_csr = csr_matrix(adj_matrix_coo) + compute_sec_adj(csr_matrix(adj_matrix_coo))  # 邻接矩阵+次级邻居
        choice_adj_coo = choice_adj_csr.tocoo()
        # 获取每一行非零元素的列索引
        # print(choice_adj_csr[high_g_node, 0])
        # arr1d_for_sort_g = np.zeros(num_nodes, dtype=int)
        unsort_col_index = choice_adj_coo.col[choice_adj_coo.row == high_g_node]

        # print(f"high_g_node {high_g_node}'s latent friend-range column index: {choice_adj_coo.col[choice_adj_coo.row == high_g_node]}")

        # for col in choice_adj_coo.col[choice_adj_coo.row == high_g_node]:
        #     arr1d_for_sort_g[col] = grades_arr[col]  # 查询分数,这是一个用来存放分数的数组，num_nodes列，所以有的列上的值会为0
        arr1d_for_sort_g = col_index_to_grade(choice_adj_coo, high_g_node)
        # 一个排序列标的函数，把这些列索引都映成分数，再返回排序后分数的列索引
        # for row, col, value in zip(choice_adj_coo.row, choice_adj_coo.col, choice_adj_coo.data):
        #     if row == high_g_node:
        #         arr1d_for_sort_g[col] = grades_arr[col]
        #           print(f"Column {col}: {value}")

        # print("arr1d_for_sort_g:", arr1d_for_sort_g)
        # latent friend-range 按照分数从高到低排序后的列索引
        nonzero_count_in_g = np.count_nonzero(arr1d_for_sort_g)
        sorted_col_index = np.argsort(arr1d_for_sort_g)[::-1][:nonzero_count_in_g]  # 这是从高到低分数排序的列索引，只保留有效分数的列索引
        # print("sorted_col_index:", sorted_col_index)
        new_degree = 0  # 校验选择者在他的preference里只能有效的执行degree次
        for t_col in range(nonzero_count_in_g):  # high_g_node遍历了他的choice_adj_coo里的所有人
            if new_degree < degree:
                # 判断现在即将成为朋友的人，他不是high_g_node已经有的朋友
                if sorted_col_index[t_col] in adj_matrix_coo.col[adj_matrix_coo.row == high_g_node]:
                    new_degree += 1
                    continue
                else:
                    pre_f_col = sorted_col_index[t_col]
                    # 这是一次双向选择，选择者的分数要至少大于被选者朋友中任何一人的分数，没有等号表示若分数相同，默认保留旧的朋友
                    pre_f_col_g = col_index_to_grade(adj_matrix_coo, pre_f_col)
                    if np.any(grades_arr[high_g_node] > pre_f_col_g[np.nonzero(pre_f_col_g)]):
                        drop_the_lowest_g(adj_matrix_coo, pre_f_col)
                        drop_the_lowest_g(adj_matrix_coo, high_g_node)  # 成为朋友的两个人各自都要drop掉一个现有的朋友，始终保持朋友数量不变
                        modify_adj_m_to_one(adj_matrix_coo, high_g_node, pre_f_col)
                        modify_adj_m_to_one(adj_matrix_coo, pre_f_col, high_g_node)
                        new_degree += 1
                    else:
                        continue

        print(f"high_g_node number {high_g_node} has finished his choosing")
        #  help the DRAGS

        arr1d_g_f_sort_of_lack_matrix = np.zeros(num_nodes, dtype=int)  # 存储列索引映成分数的一维数组，只用作排序
        for t1_col in range(num_nodes):  # 把列索引映成分数的循环
            if lack_of_f_matrix[t1_col] < 0:
                arr1d_g_f_sort_of_lack_matrix[t1_col] = grades_arr[t1_col]

        sorted_col_index_of_lack_m = np.argsort(arr1d_g_f_sort_of_lack_matrix)[::-1]  # 从高到低分数排序的列索引
        sorted_col_index_of_lack_m = sorted_col_index_of_lack_m[
                                     :np.count_nonzero(lack_of_f_matrix < 0)]  # 只保留前面真正drag的列索引
        if sorted_col_index_of_lack_m.shape[0] < 2:  # 避免f_choices为空
            # print(f"len of sorted_col_index_of_lack_m is {sorted_col_index_of_lack_m.shape[0]}, continue")
            continue
        f_choices = sorted_col_index_of_lack_m.copy()
        # print("sorted_col_index_of_lack_m:", sorted_col_index_of_lack_m)
        # print("f_choices:", f_choices)
        for high_g_in_drags in sorted_col_index_of_lack_m:  # 在缺失朋友的人中从分数最高的人开始，为他补齐朋友，因为分数高的人会先成为high_g_node
            if lack_of_f_matrix[high_g_in_drags] < 0:  # 校验他的确还缺少朋友
                lacking_f_num = np.abs(lack_of_f_matrix[high_g_in_drags])  # t2_col这个人缺失的朋友数量
                f_choices = f_choices[1:]  # 从除去第一个人以外的范围中选择
                if len(f_choices) == 0:  # 避免drags pool 里都是熟人而导致它为空
                    continue
                elif lacking_f_num <= len(f_choices):
                    lacking_f_num_real = lacking_f_num
                else:
                    lacking_f_num_real = len(f_choices)  # 这个池子里除去第一个人以外，的数量总和
                    print("**** drags pool lacks people ***\n")
                print(
                    f"high_g_in_drags {high_g_in_drags} need {lacking_f_num} friends, he will make {lacking_f_num_real} new friends ")
                for t_times in range(lacking_f_num_real):
                    print(f"before f_choices:", f_choices)
                    # 为了保持随机性，用while遍历f_choices是为了避免当两个已经是朋友的人，同时落入drags pool时，依旧相互选择。
                    random_f_index = None
                    check_len = 0
                    f_choices_temp = f_choices.copy()  # 创建temp是为了让他强制随机到一个陌生人
                    # f_choices_tt = f_choices.copy()
                    while len(f_choices_temp) > 0:
                        random_f_index = f_choices_temp[0]  # 随机选一个
                        if get_coo_value(adj_matrix_coo, high_g_in_drags, random_f_index) == 0:  # 如果随机到的是陌生人，则停止
                            break
                        else:
                            f_choices_temp = np.delete(f_choices_temp, np.where(f_choices_temp == random_f_index))
                    # 若等号成立，则说明对于high_g_in_drags来说，drags pool里没有陌生人，也就是他无法从这里结交新的朋友
                    if len(f_choices_temp) == 0:
                        # f_choices = np.append(f_choices, high_g_in_drags)  #感觉没有必要把high_g_in_drags连接在最后
                        print(f"there is no stranger for number {high_g_in_drags} in drag pool; after f_choices:",
                              f_choices)
                        continue

                    modify_adj_m_to_one(adj_matrix_coo, high_g_in_drags, random_f_index)
                    modify_adj_m_to_one(adj_matrix_coo, random_f_index, high_g_in_drags)
                    lack_of_f_matrix[high_g_in_drags] += 1
                    lack_of_f_matrix[random_f_index] += 1  # 扶贫成功后这两个人在lack_of_f_matrix里的值要做变化
                    if lack_of_f_matrix[random_f_index] == 0:
                        # 若分数位于后面的人已经被选中，且其lack_of_f_matrix的值归零,则将其剔除，f_choices是列索引
                        f_choices = np.delete(f_choices, np.where(f_choices == random_f_index))
                    # print(f"this is printed in th t_times for circle:\n", adj_matrix_coo.toarray())

                    print(f"after f_choices:", f_choices)
'''



def col_index_to_sumf_grade(t_coo_matrix, t_row_number, grades_arr_sum_f_t):  # 此函数会把为1的元素映射成他们的sumf分数
    t_arr1d_for_sort_g = np.zeros(num_nodes, dtype=int)
    for col in t_coo_matrix.col[t_coo_matrix.row == t_row_number]:
        t_arr1d_for_sort_g[col] = grades_arr_sum_f_t[col]
    return t_arr1d_for_sort_g


def drop_the_lowest_sum_f_g(t_coo_matrix, t_row_number):  # 这个函数把接收到的这个行标的朋友里分数最低的那个人drop掉
    each_f_grade_list = col_index_to_sumf_grade(t_coo_matrix, t_row_number, grades_arr_standard)
    each_f_nonzero_count = np.count_nonzero(each_f_grade_list)  # 这是each_f_grade_list里不为零的元素的个数
    drop_col_index = np.argsort(each_f_grade_list)[-each_f_nonzero_count:]  # 这是一个分数从低高到排序后返回的列索引，只保留倒数each_f_nonzero_count个
    f_be_dropped_col_num = drop_col_index[0]  # 分数最低的那个倒霉蛋
    modify_adj_m_to_zero(t_coo_matrix, t_row_number, f_be_dropped_col_num)
    modify_adj_m_to_zero(t_coo_matrix, f_be_dropped_col_num, t_row_number)
    lack_of_f_matrix[f_be_dropped_col_num] -= 1  # 因为这个倒霉蛋（f_be_dropped_col_num），他是被迫失去了一个朋友，所以给他-1


grades_arr_sum_f_ini = np.zeros(num_nodes, dtype=int)
for i11 in range(num_nodes):
    sum_f_ini = np.sum(col_index_to_grade(adj_matrix_coo_ini, i11))  # i1这个人的朋友的分数求和
    grades_arr_sum_f_ini[i11] = sum_f_ini
grades_arr_standard_ini = grades_arr + grades_arr_sum_f_ini
grades_arr_sum_f = np.zeros(num_nodes, dtype=int)

indices_t = np.array([], dtype=int)# indices_t是分数为9,5,1的人(包含重复分数)的列索引
# t1_chart = 9
# for i2 in range(len(grades_arr)):
#     if grades_arr[i2] == t1_chart:
#         indices_t = np.append(indices_t, i2)
#         t1_chart = t1_chart - 4

arr_num = np.zeros(3, dtype=int)  # 用来存放有多少个重复的9,5,1
for i2 in range(len(sorted_indices)):
    if grades_arr[sorted_indices[i2]] == 9:
        indices_t = np.append(indices_t, sorted_indices[i2])
        arr_num[0] += 1
    elif grades_arr[sorted_indices[i2]] == 5:
        indices_t = np.append(indices_t, sorted_indices[i2])
        arr_num[1] += 1
    elif grades_arr[sorted_indices[i2]] == 1:
        indices_t = np.append(indices_t, sorted_indices[i2])
        arr_num[2] += 1
grades_sum_f_chart = np.zeros(len(indices_t), dtype=int)

print("\nthe following is refining process:\n")
# adj_draw_network(adj_matrix_coo)
# 精细演化，这次演化里让他们按照sum_f的高低来选择preference
flag = True
evo_times = 0

while flag:
    evo_times += 1
    print("evo_times:",evo_times)
    for i1 in range(num_nodes):
        sum_f = np.sum(col_index_to_grade(adj_matrix_coo, i1))  # i1这个人的朋友的分数求和
        grades_arr_sum_f[i1] = sum_f
    grades_arr_standard = 0.5*grades_arr + 0.5*grades_arr_sum_f  # 每一个人朋友的分数总和+他本人的分数，构成他的标准分数
    sorted_indices_standard = np.argsort(grades_arr_standard)[::-1]

    # grades_sum_f_chart_t[indices_t] 只取出需要折线图的几人的分数
    grades_sum_f_chart = np.vstack((grades_sum_f_chart, grades_arr_sum_f[indices_t][np.newaxis, :]))  # 反复叠加grades_arr_sum_f行，只用于画折线图

    for high_g_node in sorted_indices_standard:
        choice_adj_csr = csr_matrix(adj_matrix_coo) + compute_sec_adj(csr_matrix(adj_matrix_coo))  # 邻接矩阵+次级邻居
        choice_adj_coo = choice_adj_csr.tocoo()
        # arr1d_for_sort_g = col_index_to_grade(choice_adj_coo, high_g_node)
        # nonzero_count_in_g = np.count_nonzero(arr1d_for_sort_g)
        # sorted_col_index = np.argsort(arr1d_for_sort_g)[::-1][:nonzero_count_in_g]  # 这是从高到低分数排序的列索引，只保留有效分数的列索引

        sum_f_1d_for_sort = col_index_to_sumf_grade(choice_adj_coo, high_g_node, grades_arr_standard)
        nonzero_count1 = np.count_nonzero(sum_f_1d_for_sort)
        sorted_col_index_sum_f = np.argsort(sum_f_1d_for_sort)[::-1][:nonzero_count1]
        new_degree = 0
        for t_col in range(nonzero_count1):  # high_g_node遍历了他的choice_adj_coo里的所有人
            if new_degree < degree:
                # 判断现在即将成为朋友的人，他不是high_g_node已经有的朋友
                if sorted_col_index_sum_f[t_col] in adj_matrix_coo.col[adj_matrix_coo.row == high_g_node]:
                    new_degree += 1
                    continue
                else:
                    pre_f_col = sorted_col_index_sum_f[t_col]
                    # 这是一次双向选择，选择者的分数要至少大于被选者朋友中任何一人的分数，没有等号表示若分数相同，默认保留旧的朋友
                    pre_f_col_sum_f_g = col_index_to_sumf_grade(adj_matrix_coo, pre_f_col, grades_arr_standard)
                    if np.any(grades_arr_standard[high_g_node] > pre_f_col_sum_f_g[np.nonzero(pre_f_col_sum_f_g)]):  # 利用nonzero取出非零元素
                        drop_the_lowest_sum_f_g(adj_matrix_coo, pre_f_col)
                        drop_the_lowest_sum_f_g(adj_matrix_coo, high_g_node)  # 成为朋友的两个人各自都要drop掉一个现有的朋友，始终保持朋友数量不变
                        modify_adj_m_to_one(adj_matrix_coo, high_g_node, pre_f_col)
                        modify_adj_m_to_one(adj_matrix_coo, pre_f_col, high_g_node)
                        new_degree += 1
                    else:
                        continue

        # print(f"high_g_node number {high_g_node} has finished his choosing")
        #  help the DRAGS

        arr1d_g_f_sort_of_lack_matrix = np.zeros(num_nodes, dtype=int)  # 存储列索引映成分数的一维数组，只用作排序
        for t1_col in range(num_nodes):  # 把列索引映成分数的循环
            if lack_of_f_matrix[t1_col] < 0:
                arr1d_g_f_sort_of_lack_matrix[t1_col] = grades_arr_standard[t1_col]

        sorted_col_index_of_lack_m = np.argsort(arr1d_g_f_sort_of_lack_matrix)[::-1]  # 从高到低分数排序的列索引
        sorted_col_index_of_lack_m = sorted_col_index_of_lack_m[
                                     :np.count_nonzero(lack_of_f_matrix < 0)]  # 只保留前面真正drag的列索引
        if sorted_col_index_of_lack_m.shape[0] < 2:  # 避免f_choices为空
            # print(f"len of sorted_col_index_of_lack_m is {sorted_col_index_of_lack_m.shape[0]}, continue")
            continue
        f_choices = sorted_col_index_of_lack_m.copy()

        for high_g_in_drags in sorted_col_index_of_lack_m:  # 在缺失朋友的人中从分数最高的人开始，为他补齐朋友，因为分数高的人会先成为high_g_node
            if lack_of_f_matrix[high_g_in_drags] < 0:  # 校验他的确还缺少朋友
                lacking_f_num = np.abs(lack_of_f_matrix[high_g_in_drags])  # t2_col这个人缺失的朋友数量
                f_choices = f_choices[1:]  # 从除去第一个人以外的范围中选择
                if len(f_choices) == 0:  # 避免drags pool 里都是熟人而导致它为空
                    continue
                elif lacking_f_num <= len(f_choices):
                    lacking_f_num_real = lacking_f_num
                else:
                    lacking_f_num_real = len(f_choices)  # 这个池子里除去第一个人以外，的数量总和
                    # print("**** drags pool lacks people ***\n")
                # print(f"high_g_in_drags {high_g_in_drags} need {lacking_f_num} friends, he will make {lacking_f_num_real} new friends ")
                for t_times in range(lacking_f_num_real):
                    # print(f"before f_choices:", f_choices)
                    # 为了保持随机性，用while遍历f_choices是为了避免当两个已经是朋友的人，同时落入drags pool时，依旧相互选择。
                    random_f_index = None
                    check_len = 0
                    f_choices_temp = f_choices.copy()  # 创建temp是为了让他强制随机到一个陌生人
                    while len(f_choices_temp) > 0:
                        random_f_index = f_choices_temp[0]
                        if get_coo_value(adj_matrix_coo, high_g_in_drags, random_f_index) == 0:  # 如果随机到的是陌生人，则停止
                            break
                        else:
                            f_choices_temp = np.delete(f_choices_temp, np.where(f_choices_temp == random_f_index))
                    # 若等号成立，则说明对于high_g_in_drags来说，drags pool里没有陌生人，也就是他无法从这里结交新的朋友
                    if len(f_choices_temp) == 0:
                        # f_choices = np.append(f_choices, high_g_in_drags)  #感觉没有必要把high_g_in_drags连接在最后
                        # print(f"there is no stranger for number {high_g_in_drags} in drag pool; after f_choices:",f_choices)
                        continue

                    modify_adj_m_to_one(adj_matrix_coo, high_g_in_drags, random_f_index)
                    modify_adj_m_to_one(adj_matrix_coo, random_f_index, high_g_in_drags)
                    lack_of_f_matrix[high_g_in_drags] += 1
                    lack_of_f_matrix[random_f_index] += 1  # 扶贫成功后这两个人在lack_of_f_matrix里的值要做变化
                    if lack_of_f_matrix[random_f_index] == 0:
                        # 若分数位于后面的人已经被选中，且其lack_of_f_matrix的值归零,则将其剔除，f_choices是列索引
                        f_choices = np.delete(f_choices, np.where(f_choices == random_f_index))
                    # print(f"this is printed in th t_times for circle:\n", adj_matrix_coo.toarray())

                    # print(f"after f_choices:", f_choices)

    # grades_sum_f_chart = np.vstack((grades_sum_f_chart, grades_arr_sum_f[indices_t]))


    last_two_rows = grades_sum_f_chart[-2:]
    c = 0
    for i1 in range(num_nodes):
        len_each_row = len(adj_matrix_coo.col[adj_matrix_coo.row == i1])
        if len_each_row == degree:
            c += 1
    if c == num_nodes and np.all(last_two_rows[0] == last_two_rows[1]):
        flag = False
    # if c == num_nodes:





print("\nchoosing process finished")
grades_arr_standard = 0.5*grades_arr + 0.5*grades_arr_sum_f
grades_sum_f_chart = grades_sum_f_chart[1:]  # 删除第一行
# print("grades arr", grades_arr)
# print("adj_to_grade(adj_matrix_coo):\n", adj_to_grade(adj_matrix_coo_ini))
# print("adj_to_grade(adj_matrix_coo):\n", adj_to_grade(adj_matrix_coo))

# print("adj_matrix_coo:\n", adj_matrix_coo.toarray())
# plot_histogram(grades_arr)
# plot_heatmap(adj_to_grade(adj_matrix_coo_ini), grades_arr_standard_ini)

plot_heatmap(adj_to_grade(adj_matrix_coo), grades_arr_standard)
# arr1 = col_index_to_grade(adj_matrix_coo, indices_t[0])
# print("grade 1 's friends: ", arr3[np.nonzero(arr3)])



# line_chart(grades_sum_f_chart, arr_num, evo_times)
adj_draw_network(adj_matrix_coo, evo_times)

# c = 0
# for i1 in range(num_nodes):
#     len_each_row = len(adj_matrix_coo.col[adj_matrix_coo.row == i1])
#     if len_each_row == degree:
#         c += 1
# if c == num_nodes:
#     print("finish")


'''
c = 0
c1 = 0
for i1 in range(num_nodes):
    len_each_row = len(adj_matrix_coo.col[adj_matrix_coo.row == i1])  # 提取一行作为一维数组
    if len_each_row == degree:
        c += 1
    elif len_each_row > degree:
        c1 += 1
    else:
        print(f"number {i1} is lack of f")
if c == num_nodes:
    print("\nchoosing process finished")
else:
    print(f"\nNahhhh, remaining {num_nodes - c - c1} rows smaller than degree ,{c1} bigger than degree")
'''

# 记录结束时间
end_time = time.time()
# 计算代码执行时间
execution_time = end_time - start_time
execution_time = round(execution_time/60.0, 2)


# 打印代码执行时间
print(f"execute time：{round(execution_time, 2)} seconds, {round(execution_time/60.0, 2)} minutes")

