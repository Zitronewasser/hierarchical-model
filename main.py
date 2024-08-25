import time
from statistic import adj_draw_network,plot_histogram
from generateData import adj_matrix_coo_ini, grades_arr
# 记录开始时间
start_time = time.time()
plot_histogram(grades_arr)
# adj_draw_network(adj_matrix_coo_ini, 0)

# 在这里写下您的代码


# 记录结束时间
end_time = time.time()
# 计算代码执行时间
execution_time = end_time - start_time
# 打印代码执行时间
print("execute time：", round(execution_time, 2), "seconds")


