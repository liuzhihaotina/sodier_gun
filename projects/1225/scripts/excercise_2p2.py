import pandas as pd
import torch

# 0) 创建“更多行和列”的原始数据集（包含数值列、类别列、缺失值）
raw = pd.DataFrame({
    "NumRooms":   [3, 2, None, 4, 3, 5, None, 2, 4, 3],
    "Alley":      ["Pave", None, None, "Grvl", "Pave", None, "Grvl", None, "Pave", None],
    "YearBuilt":  [2000, 1995, 2010, None, 2005, 1999, 2012, None, 2001, 2008],
    "Pool":       [None, None, None, None, None, "Y", None, None, None, None],  # 缺失最多的列(大部分都是 NaN)
    "Area":       [80, 60, 75, 90, None, 120, 70, 65, None, 85],
    "Price":      [300, 220, 260, 400, 330, 520, 280, 210, 350, 360],  # 标签(输出)
})

print("原始数据：")
print(raw)
print()

# 1) 删除缺失值最多的列（只在特征列里找；不动标签列 Price）
inputs = raw.drop(columns=["Price"])
outputs = raw["Price"]

missing_counts = inputs.isna().sum()                # 每列缺失值数量
col_most_missing = missing_counts.idxmax()          # 缺失最多的列名（如有并列，取第一个）
inputs = inputs.drop(columns=[col_most_missing])    # 删除该列

print("每列缺失值数量：")
print(missing_counts)
print(f"\n删除缺失最多的列: {col_most_missing}\n")

print("删除后的 inputs：")
print(inputs)
print()

# 2) 预处理：数值列用均值填充；类别列 one-hot（并把 NaN 也当一个类别）
num_cols = inputs.select_dtypes(include="number").columns
inputs[num_cols] = inputs[num_cols].fillna(inputs[num_cols].mean())

inputs = pd.get_dummies(inputs, dummy_na=True)  # 类别列变成 0/1；NaN 也单独成列

print("预处理后的 inputs（全是数值列）：")
print(inputs)
print()

# 3) 将预处理后的数据集转换为张量格式（tensor）
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float)).reshape(-1, 1)

print("X tensor shape:", X.shape)
print("y tensor shape:", y.shape)
print("X dtype:", X.dtype, "y dtype:", y.dtype)

# 可选：看一下张量内容
print("\nX tensor:\n", X)
print("\ny tensor:\n", y)
