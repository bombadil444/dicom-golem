import csv
import json

def process():
    p = open('/golem/work/params.json')
    params = json.load(p)
    partition = int(params['partition'])
    step = int(params['step'])
    column = int(params['column'])
    search = params['search']

    minimum = partition * step
    maximum = minimum + step

    with open('/golem/resource/example.csv') as data:
        csv_reader = csv.reader(data, delimiter=",")
        with open(f'/golem/output/output_{partition}.txt', 'x') as output:
            for i, row in enumerate(csv_reader):
                if i > minimum and i < maximum:
                    if row[column] == search:
                        output.write("found")

if __name__ == "__main__":
    process()	
