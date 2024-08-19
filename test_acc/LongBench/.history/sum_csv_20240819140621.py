import os
import sys
import pandas as pd
import json

path = sys.argv[1]
model = path
if model[-1] == '/':
    model = model[:-1]
model = model.split('/')[-1]

df = pd.DataFrame(columns=["task", "score", "model", "method"])
if path[-1] != '/':
    path = path + '/'
for root, dirs, files in os.walk(path):
    for dir in dirs:
        if dir[-1] != '/':
            dir = dir + '/'
        results_path = path + dir + 'results.json'
        method = dir
        if method[-1] == '/':
            method = method[:-1]
        method = method.split('/')[-1]
        with open(results_path) as f:
            results = json.load(f)

        for task in results:
            df = pd.concat([df, pd.DataFrame([[task, results[task], model, method]], columns=["task", "score", "model", "method"])], ignore_index=True)

df.to_csv(path + 'results.csv', index=False)





        
    

        