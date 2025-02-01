import os
import sys

path = sys.argv[1]
for root, dirs, files in os.walk(path):
    for dir in dirs:
        if path[-1] != '/':
            path = path + '/'
        if dir[-1] != '/':
            dir = dir + '/'
        os.system("python src/eval.py --path " + path + dir)

