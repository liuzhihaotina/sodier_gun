# from build import soldier_gun

# s=soldier_gun.Soldier("liuzhihao")
# g=soldier_gun.Gun("AK47")
# tmp_count1=g._bullet_count

# s.addGun(g)
# s.addBulletToGun(20)
# tmp_count2=g._bullet_count
# s.fire()
# print(f"There is one soldier named {s._name}.")
# print(f"We give her one {g._type} which has {tmp_count1} bullet(s).")
# print(f"g._bullet_count={g._bullet_count}，tmp_count={tmp_count1}")
# print(f"She adds {tmp_count2-tmp_count1} bullet(s) to it.")
# print(f"After she fires the gun, it has {g._bullet_count} bullert(s) left.")


from build import draw

draw.drw()

# import sys
# import os

# build_path = '/home/tina/workspaces/soldiers_guns/build'
# sys.path.append(build_path)

# # 检查目录内容
# print("Build 目录内容:", os.listdir(build_path))

# # 查找 .so 文件
# for file in os.listdir(build_path):
#     if file.endswith('.so'):
#         print(f"找到 .so 文件: {file}")

# # 尝试导入
# try:
#     import draw
#     print("导入成功!")
#     draw.drw()
# except ImportError as e:
#     print(f"导入失败: {e}")
