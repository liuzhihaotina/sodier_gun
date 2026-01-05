# import pandas as pd

# data_file = 'projects/1225/data/house_tiny.csv'
# data = pd.read_csv(data_file)
# print(data)

# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# print(inputs)
import torch
import pandas as pd

data_file = 'projects/1225/data/house_tiny2.csv'
data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:, 0:3], data.iloc[:, 3]

# 只选择数值列
num_cols = inputs.select_dtypes(include='number').columns
inputs[num_cols] = inputs[num_cols].fillna(inputs[num_cols].mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(inputs)
