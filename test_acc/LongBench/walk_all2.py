import os

def run_eval_and_sum(base_dir):
    # 用于存储一级目录的路径，以便之后执行 sum_csv.py
    first_level_dirs = set()

    # 遍历第一层目录
    for root, dirs, files in os.walk(base_dir):
        for sub_dir in dirs:
            # 构建第二层目录的路径
            second_level_dir = os.path.join(root, sub_dir)

            # 遍历第二层目录
            for second_root, second_dirs, second_files in os.walk(second_level_dir):
                relative_path = os.path.relpath(second_root, base_dir)

                # 如果是二级目录，执行 eval.py
                if len(relative_path.split(os.sep)) == 2:
                    print(f"正在运行 eval.py: {second_root}")
                    os.system(f"python src/eval.py --path {second_root}/")

                    # 添加一级目录路径以便之后执行 sum_csv.py
                    first_level_dirs.add(root)

                # 进入二级目录后停止继续递归
                break

    # 对所有一级目录执行 sum_csv.py
    for first_level_dir in first_level_dirs:
        print(f"正在运行 sum_csv.py: {first_level_dir}")
        os.system(f"python sum_csv.py {first_level_dir}/")

# 基础路径
base_directory = './results/'

# 运行函数
run_eval_and_sum(base_directory)
