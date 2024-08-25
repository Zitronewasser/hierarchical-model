import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv('save_file/combined_vectors.csv')

# 提取列名
columns = data.columns

# 设置Seaborn的风格和颜色
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(data['Index'], data[columns[1]], marker='o', label='average class-score', color=palette[0])
plt.plot(data['Index'], data[columns[2]], marker='o', label='mean friends-score', color=palette[1])
plt.plot(data['Index'], data[columns[3]], marker='o', label='min friends-score', color=palette[2])

# 填充区域
plt.fill_between(data['Index'], data[columns[1]], data[columns[2]], color=palette[0], alpha=0.3)
plt.fill_between(data['Index'], data[columns[2]], data[columns[3]], color=palette[1], alpha=0.3)

# 设置标题和标签
plt.title('Evolution of Node Ability and Friends\' Ability', fontsize=16)
plt.xlabel('Evolving steps', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.legend()

# 显示图表
plt.show()
plt.savefig('save_file/evolution_of_node_ability_and_friends_ability.png')
