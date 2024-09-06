import os

def find_missing_files(base_dir, required_files):
    missing_files = {}

    # 遍历第一层目录
    for root, dirs, files in os.walk(base_dir):
        for sub_dir in dirs:
            # 构建第二层目录的路径
            second_level_dir = os.path.join(root, sub_dir)

            # 遍历第二层目录
            for second_root, second_dirs, second_files in os.walk(second_level_dir):
                relative_path = os.path.relpath(second_root, base_dir)
                # 检查是否是二级目录下的文件
                if len(relative_path.split(os.sep)) == 2:
                    folder_missing = [file for file in required_files if file not in second_files]
                    if folder_missing:
                        missing_files[relative_path] = folder_missing

                # 一旦进入二级目录，停止继续递归
                break

    return missing_files


# 基础路径和必须存在的文件列表
base_directory = './'
required_files = ['2wikimqa.jsonl', 'dureader.jsonl', 'gov_report.jsonl', 'hotpotqa.jsonl', 'lcc.jsonl', 'lsht.jsonl', 'multifieldqa_en.jsonl', 'multifieldqa_zh.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'narrativeqa.jsonl', 'passage_count.jsonl', 'passage_retrieval_en.jsonl', 'passage_retrieval_zh.jsonl', 'qasper.jsonl', 'qmsum.jsonl', 'repobench-p.jsonl', 'samsum.jsonl', 'trec.jsonl', 'triviaqa.jsonl', 'vcsum.jsonl']

missing = find_missing_files(base_directory, required_files)

# 输出缺失文件的文件夹和文件名
if missing:
    print("以下文件缺失:")
    for folder, files in missing.items():
        print(f"文件夹: {folder}, 缺失文件: {files}")
else:
    print("所有文件齐全。")
