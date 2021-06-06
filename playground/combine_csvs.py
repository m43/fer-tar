import csv
import os
from collections import namedtuple

path_to_folder_with_csvs = "/home/user72/Desktop/csvs"
path_to_output_csv = "/home/user72/Desktop/combined.csv"

assert os.path.isdir(path_to_folder_with_csvs)

csv_file_tuple = namedtuple('CSVFileTuple', ['path', 'name'])
csv_file_tuples = []
for filename in sorted(os.listdir(path_to_folder_with_csvs)):
    if not filename.endswith(".csv"):
        continue
    t = csv_file_tuple(os.path.join(path_to_folder_with_csvs, filename), filename[:-4])
    csv_file_tuples.append(t)

results = []
for path, name in csv_file_tuples:
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            if path == csv_file_tuples[0][0]:
                # Add the first 4 rows from the first csv
                if i == 0:
                    results.append(row[0:4] + ["ae"])
                else:
                    results.append(row[0:4] + ["mean"])
                    results.append(row[0:4] + ["std"])

            if i == 0:
                # Header
                results[i].append(name.replace(",", ";"))
            else:
                # Metrics
                results[2*i-1].append(row[4])
                results[2*i].append(row[5])

            assert len(results[0]) == len(results[i])

with open(path_to_output_csv, 'w') as f:
    for row in results:
        line = ",".join(row)
        f.write(line + "\n")
        print(line)
