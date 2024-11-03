


'''
重命名image文件
'''

import os

# 指定文件夹路径
folder_path = r"D:\BaiduNetdiskDownload\UCF-18\image\Test"

# 获取文件夹下所有文件
files = os.listdir(folder_path)

# 过滤出图片文件，这里假设图片格式为.jpg，您可以根据需要修改
image_files = [f for f in files if f.endswith('.jpg')]

# 对图片文件进行排序，以确保按照正确的顺序重命名
image_files.sort()

# 初始化重命名的起始编号
rename_count = 1

# 遍历图片文件并重命名
for file in image_files:
    # 构建原文件路径和新文件路径
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, f"{rename_count}.jpg")

    # 重命名文件
    os.rename(old_file_path, new_file_path)

    # 打印重命名信息
    print(f"Renamed {old_file_path} to {new_file_path}")

    # 更新重命名的编号
    rename_count += 1

print("All files have been renamed successfully.")



'''
重命名txt文件
'''

import os

# 指定文件夹路径
folder_path = r"D:\BaiduNetdiskDownload\UCF-18\label\Test"

# 获取文件夹下所有文件
files = os.listdir(folder_path)

# 过滤出txt文件
txt_files = [f for f in files if f.endswith('.txt')]

# 对txt文件进行排序，以确保按照正确的顺序重命名
txt_files.sort()

# 初始化重命名的起始编号
rename_count = 1

# 遍历txt文件并重命名
for file in txt_files:
    # 构建原文件路径和新文件路径
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, f"{rename_count}.txt")

    # 重命名文件
    os.rename(old_file_path, new_file_path)

    # 打印重命名信息
    print(f"Renamed {old_file_path} to {new_file_path}")

    # 更新重命名的编号
    rename_count += 1

print("All label files have been renamed successfully.")