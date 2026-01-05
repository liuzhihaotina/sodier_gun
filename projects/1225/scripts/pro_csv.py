import os

data_dir = 'projects/1225'
os.makedirs(os.path.join(data_dir, 'data'), exist_ok=True)
data_file = os.path.join(data_dir, 'data', 'house_tiny2.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Num,Price\n') # 列名
    f.write('NA,Pave,3,127500\n') # 每行表示一个数据样本
    f.write('2,NA,NA,106000\n')
    f.write('4,NA,5,178100\n')
    f.write('NA,NA,NA,140000\n')