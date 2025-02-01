import os

def run_eval_and_sum(base_dir):
    first_level_dirs = set()

    for root, dirs, files in os.walk(base_dir):
        for sub_dir in dirs:
            second_level_dir = os.path.join(root, sub_dir)
            for second_root, second_dirs, second_files in os.walk(second_level_dir):
                relative_path = os.path.relpath(second_root, base_dir)

                if len(relative_path.split(os.sep)) == 2:
                    print(f"running eval.py: {second_root}")
                    os.system(f"python src/eval.py --path {second_root}/")

                    first_level_dirs.add(root)

                break

    for first_level_dir in first_level_dirs:
        print(f"running sum_csv.py: {first_level_dir}")
        os.system(f"python sum_csv.py {first_level_dir}/")

base_directory = './results/'

run_eval_and_sum(base_directory)
