import pickle

import numpy as np
import pandas as pd

# 定义文件路径
files = {
    'vector_list_2d_self_ss': 'save_file/vector_list_2d_self.pkl',
    'vector_list_2d_min': 'save_file/vector_list_2d_min.pkl',
    'vector_list_2d_mean': 'save_file/vector_list_2d_mean.pkl'
}

# 读取每个文件并存储到一个字典中
data = {}
for name, path in files.items():
    with open(path, 'rb') as file:
        data[name] = pickle.load(file)

# 创建一个空的DataFrame
df = pd.DataFrame()
max_length = max(len(v) for v in data.values())
df['Index'] = range(1, max_length + 1)
# 将每个文件的内容添加到DataFrame中，每个文件作为单独的一列
for name, vector_list in data.items():

    flattened_list = [round(np.mean(item[0]), 4) for item in vector_list]
    df[name] = flattened_list
# 保存到CSV文件
output_file = 'save_file/combined_vectors.csv'
df.to_csv(output_file, index=False)

print(f'Data saved to {output_file}')
