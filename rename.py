import os

def rename_files_with_prefix(folder_path, prefix, new_prefix):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            # 提取序号部分
            index = filename[len(prefix):]

            # 构造新的文件名
            new_filename = new_prefix + index

            # 构造原文件路径和新文件路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)

# 示例用法
folder_path = '/root/docqa/document_intelligence/doc_vqa/OCR_process/demo_pics/minivision'  # 文件夹路径
prefix = 'old_'  # 原始文件名的前缀
new_prefix = 'new_'  # 新文件名的前缀
rename_files_with_prefix(folder_path, prefix, new_prefix)
