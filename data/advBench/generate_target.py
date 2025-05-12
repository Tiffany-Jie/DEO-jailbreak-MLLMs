import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('harmful_behaviors.csv')

# 提取target列
target = df['target']

# 将target列保存到新的CSV文件
target.to_csv('target.csv', index=False)
