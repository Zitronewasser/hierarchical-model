from matrixA import SumP, PIHouse, NumHouse, A
import numpy as np
from statistic import plot_heatmap

# # 打印结果
# for i, arr in enumerate(A):
#     print(f"the {i+1} row of matrix A is: {arr}")
#
# # 调用这个矩阵A的某一个数
# print(f"The first number of 20 row in matrix A is: {A[19][0]}")

# 先生成一个3栋楼，每栋楼里3个人的初始化relation matrix，用二维数组的形式表示
# array_R = [[0] * SumP for _ in range(SumP)]
array_R = np.zeros((SumP, SumP), dtype=int)

for k in range(NumHouse):
    for i in range(int((k - 1) * PIHouse), PIHouse + int((k - 1) * PIHouse)):
        for j in range(int((k - 1) * PIHouse), PIHouse + int((k - 1) * PIHouse)):
            array_R[i][j] = 1


# for k in range(NumHouse):
#     if k == NumHouse:  # 要使得尾首相连
#         array_R[int((k - 1) * PIHouse)][0] = -1
#         array_R[0][int((k - 1) * PIHouse)] = -1 # 做一次转置即可使得对方的朋友圈也发生变化
#     else:
#         array_R[int((k - 1) * PIHouse)][int(k * PIHouse)] = -1
#         array_R[int(k * PIHouse)][int((k - 1) * PIHouse)] = -1

def grade_calculate(t_col):  # 在初始生成的分数值矩阵A中查询第t_col个人的分数
    g_row = int(t_col // PIHouse)
    g_col = int(t_col % PIHouse)  # 逆推出那个朋友在分数表中的行列，查询他的分数
    # print(f"g_row is{g_row},g_col is {g_col}")
    return A[g_row, g_col]


start = 0  # 等差数列的起始值，注意数组的第一个值的索引号为0，故起始值为0
stop = SumP  # 等差数列的结束值（不包含），总人数SumP
step = PIHouse  # 等差数列的步长，每栋房的人数PIHouse

# 生成等差数组，只利用每栋楼房的第一个人来做
FPIHouse = np.arange(start, stop, step)


# 定义了一个relation matrix(arr) to grades matrix 函数
def rm2gm(temp_a):
    fc_row = NumHouse
    fc_col = PIHouse
    _, friend_proxy_idx = np.unique(temp_a, return_index=True, axis=0)
    # 创建一个空的二维数组,这是一个用来存储新的朋友圈值的矩阵，这个矩阵里存储每栋楼第一个人的朋友的分数值
    # 这是一个NumHouse*PIHouse=SumP的矩阵
    # newArray_R = [[0] * fc_col for _ in range(fc_row)]
    newArray_R = np.zeros((fc_row, fc_col), dtype=int)
    newFC_row = 0

    for NRow in friend_proxy_idx:
        # print(f"the {NRow+1} row of array_R is:", array_R[NRow])
        newFC_col = 0
        for NCol in range(SumP):

            # print(f"the {NCol + 1} column in {NRow + 1} row of array_R is:",
            #       array_R[NRow][NCol])
            if temp_a[NRow][NCol] == 1:
                # print(f"Person {NRow + 1} and person {NCol + 1} are friends.")
                # print(f"Person {NCol + 1} 's grade is :", A[f_row][f_col], f"f_row is {f_row} f_col is {f_col}")
                # print(f"NRow is : {NRow}；NCol is : {NCol};f_row is {f_row} f_col is {f_col}")
                newArray_R[newFC_row][newFC_col] = grade_calculate(NCol)
                newFC_col += 1
        newFC_row += 1
    return newArray_R


# for i, arr in enumerate(A):
#     print(f"the {i + 1} row of matrix A is: {arr}")

# for row in newArray_R:
#     print(row)


def find_max_grade(num_p1, tu_arr1, temp_fm_g):  # 找到一个人朋友圈中分数最高的那个人的分数
    max_g = 0
    max_col_tag = num_p1
    for fm_col in range(SumP):  # 关系矩阵某一行i和列i的意义是一样的，都表示这个人i的朋友圈
        if tu_arr1[num_p1, fm_col] == 1 and fm_col != num_p1 and temp_fm_g[fm_col] == 1:  # 确定朋友的编号
            if grade_calculate(fm_col) >= max_g:  # 找分数最大的朋友
                max_g = grade_calculate(fm_col)
                max_col_tag = fm_col
    # print(f"the highest score in {num_p1}'s friends is max_g: {max_g}")
    return max_g, max_col_tag


def find_max_grade_for_p(num_p1, tu_arr1, t_preference):  # 找到一个人朋友圈中分数最高的那个人的分数
    max_g = 0
    max_col_tag = num_p1
    for fm_col in range(SumP):  # 关系矩阵某一行i和列i的意义是一样的，都表示这个人i的朋友圈
        if tu_arr1[num_p1][fm_col] == 1 and fm_col != num_p1:  # 确定朋友的编号
            if grade_calculate(fm_col) > max_g and t_preference[num_p1, fm_col] != -3:  # 找分数最大的朋友
                max_g = grade_calculate(fm_col)
                max_col_tag = fm_col
    # print(f"the highest score in {num_p1}'s friends is max_g: {max_g}")
    return max_g, max_col_tag


# create new buffer zone with -3,这里面包含不对称进入缓冲区的人和出生时的朋友

# for cb_row in range(SumP):
#     for cb_col in range(SumP):
#         if array_R[cb_row, cb_col] == 1 or buffer_zone[cb_row, cb_col] == -3:
#             new_buffer_zone[cb_row, cb_col] = -3


preference_arr = np.zeros((SumP, SumP), dtype=int)
all_one_arr_1 = np.ones((SumP, SumP), dtype=int)

being_friend_tag = np.ones(SumP, dtype=int)  # 标签数组，若还没找到PIHouse个朋友为1，否则为0


# several_high_g_tag = np.zeros((1, PIHouse), dtype=int)


def find_preference(t_preference_arr, temp_bf_tag):
    # for tp_col in range(SumP):
    #     if t_nb_zone[tp_row, tp_col] == -3 and grade_calculate(tp_col)
    all_one_arr = np.ones((SumP, SumP), dtype=int)
    for tp_row in range(SumP):
        if temp_bf_tag[tp_row] == 1:
            for tp_time in range(PIHouse - 1):  # 一栋楼里PIHouse个人，每个人最多只有PIHouse-1个朋友
                _, h_f_tag = find_max_grade(tp_row, all_one_arr, temp_bf_tag)
                all_one_arr[tp_row, h_f_tag] = 0
                t_preference_arr[tp_row, h_f_tag] = -2


def find_ones_with_high_g(t_bf_tag):
    high_g_arr = np.zeros(PIHouse, dtype=int)
    max_col_tag = 0
    max_tag_arr = np.ones(SumP, dtype=int)
    for t_fh in range(PIHouse):  # 找到还没交满朋友且分数最高的那几个人
        max_g = 0
        for fm_col in range(SumP):
            if t_bf_tag[fm_col] == 1 and max_tag_arr[fm_col]:  # 校验是否为1，是则说明他还没有交满朋友
                if grade_calculate(fm_col) >= max_g:  # 找分数最大的
                    max_g = grade_calculate(fm_col)
                    max_col_tag = fm_col
        max_tag_arr[max_col_tag] = 0
        high_g_arr[t_fh] = max_col_tag
    return high_g_arr


# being_friend_tag = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1])
# print("being_friend_tag:", being_friend_tag)
test_update_arr = np.eye(SumP, dtype=int)
check_upper_limit = np.ones(SumP, dtype=int)
# print("find_ones_with_high_g:\n", find_ones_with_high_g(being_friend_tag))


# 假定每个人都认识所有人后，直接列PL结交
for t_match in range(NumHouse):
    find_preference(preference_arr, being_friend_tag)
    for cp_row in find_ones_with_high_g(being_friend_tag):
        for cp_col in range(SumP):
            if preference_arr[cp_row][cp_col] == -2 and preference_arr[cp_col][cp_row] == -2 \
                    and check_upper_limit[cp_row] <= PIHouse - 1 \
                    and check_upper_limit[cp_col] <= PIHouse - 1:  # 限制每个人只能有PIHouse-1个朋友
                test_update_arr[cp_row][cp_col] = 1
                test_update_arr[cp_col][cp_row] = 1
                check_upper_limit[cp_row] += 1
                check_upper_limit[cp_col] += 1
        being_friend_tag[cp_row] = 0

# print("preference_arr:\n")
# for ur in preference_arr:
#     print(ur)

# print("test_update_arr:\n")
# for ur in test_update_arr:
#     print(ur)


# print(A)

t_arr = rm2gm(test_update_arr)

# ******************************************新编





