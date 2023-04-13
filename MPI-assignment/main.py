import json
import re
import time
from collections import Counter

from mpi4py import MPI
from tabulate import tabulate

TWITTER_FILE_PATH = 'smallTwitter.json'

with open('sal.json', 'r') as f:
    SAL_DATA = json.load(f)

def output_task1(counters):
    c = Counter()
    for counter in counters:
        c.update(counter)
    top_10 = c.most_common(10)
    # output the top 10 tweeters
    ranked_top_10 = [("#"+str(i),) + element for i,
                     element in enumerate(top_10, start=1)]
    headers = ["Rank", "Author Id", "Number of Tweets Made"]
    table = tabulate(ranked_top_10, headers, tablefmt='plain',
                     numalign='left', stralign='left')
    print('==================== Task 1 ====================')
    print(table)


def output_task2(counters):
    c = Counter()
    for counter in counters:
        c.update(counter)
    gcc_rank = c.most_common()

    def gcc2name(gcc):
        if gcc.startswith('1'):
            return 'Greater Sydney'
        elif gcc.startswith('2'):
            return 'Greater Melbourne'
        elif gcc.startswith('3'):
            return 'Greater Brisbane'
        elif gcc.startswith('4'):
            return 'Greater Adelaide'
        elif gcc.startswith('5'):
            return 'Greater Perth'
        elif gcc.startswith('6'):
            return 'Greater Hobart'
        elif gcc.startswith('7'):
            return 'Greater Darwin'
        elif gcc.startswith('8'):
            return 'Greater Canberra'
        elif gcc.startswith('9'):
            return 'Other Territories'

    for i in range(len(gcc_rank)):
        gcc = gcc_rank[i][0]
        gcc_rank[i] = (f'{gcc} ({gcc2name(gcc)})', gcc_rank[i][1])
    headers = ["Greater Capital City", "Number of Tweets Made"]
    table = tabulate(gcc_rank, headers, tablefmt='plain',
                     numalign='left', stralign='left')
    print('==================== Task 2 ====================')
    print(table)


def output_task3(data):
    all_author_gccs = {}
    all_author_counter = Counter()
    for i in range(len(data)):
        author_gcc = data[i][0]
        for key in author_gcc:
            if key not in all_author_gccs:
                all_author_gccs[key] = author_gcc[key]
            else:
                all_author_gccs[key].update(author_gcc[key])
        
        author_counter = data[i][1]
        all_author_counter.update(author_counter)
    rank10 = [ac[0] for ac in all_author_counter.most_common(10)]
    table = [[r, all_author_gccs[r]] for r in rank10]
    table.sort(key=lambda x: len(x[1]), reverse=True)
    for i in range(len(table)):
        table[i].insert(0, f"#{i+1}")
        table[i][2] = f"{len(table[i][2])} (#{all_author_counter[table[i][1]]} - #{', #'.join(table[i][2])})"
    header = ["Rank", "Author Id", "Number of Unique City Locations and #Tweets"]
    table = tabulate(table, header, tablefmt='plain', numalign='left', stralign='left')
    print('==================== Task 3 ====================')
    print(table)


def process_master(comm):
    start_time = time.time()

    K = comm.Get_size() - 1
    print('K:', K)

    # Open the JSON file and read its contents
    with open(TWITTER_FILE_PATH, 'r') as f:
        total_size = f.seek(0, 2)
    # print('total_size:', total_size)

    if K  == 0:
        res = [run(0, total_size, 1)]
    else:
        # Send each chunk to the corresponding slave process
        for i in range(1, K + 1):
            comm.send({
                'index': i - 1,
                'size': total_size,
                'K': K,
            }, dest=i)

        # Receive the sorted counts from each slave process and combine them
        res = []
        for i in range(1, K + 1):
            res += comm.recv(source=i)

    # gather data from all slaves
    task1_output = []
    task2_output = []
    task3_output = []
    for item in res:
        task1_output.append(item[0])
        task2_output.append(item[1])
        task3_output.append(item[2])
    output_task1(task1_output)
    output_task2(task2_output)
    output_task3(task3_output)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def get_gcc(full_name):
    # extract gcc
    city_name = full_name.lower()
    if ',' in full_name:
        city_name = full_name.split(',')[0]
    if city_name in SAL_DATA:
        return SAL_DATA[city_name]['gcc']
    else:
        return None


def process_slave(comm):
    # Receive the chunk of data to process
    data = comm.recv(source=0)
    index = data['index']
    total_size = data['size']
    K = data['K']

    res = run(index, total_size, K)
    
    comm.send([res], dest=0)


def run(index, total_size, K):
    pos = 0
    if index > 0:
        pos = index * total_size // K
    f = open(TWITTER_FILE_PATH, 'r')
    f.seek(pos)

    def get_line_value(line):
        line = line.strip()
        line = line.replace('"', '')
        if line[-1] == ',':
            line = line[:-1]
        val = line.split(': ')[1]
        return val
    
    # For task1
    id_counter = Counter()
    # For task2
    gcc_counter = Counter()
    # For task3
    author_gcc = {}
    author_unique_counter = Counter()
    author_total_counter = Counter()

    # read file by line
    while True:
        if f.tell() > (index + 1) * total_size // K:
            break
        line = f.readline()
        while line and 'author_id' not in line:
            line = f.readline()
        if not line:
            break
        author_id = get_line_value(line)

        line = f.readline()
        while line and 'full_name' not in line:
            line = f.readline()
        if not line:
            break
        full_name = get_line_value(line)
        gcc = get_gcc(full_name)

        # For task1
        id_counter[author_id] += 1

        # For task2
        if gcc is None:
            continue
        gcc_counter[gcc] += 1
        
        # For task3
        if author_id not in author_gcc:
            author_gcc[author_id] = set()
        
        author_gcc[author_id].add(gcc)
        author_total_counter[author_id] += 1
        author_unique_counter[author_id] = len(author_gcc[author_id])

    f.close()
    return [id_counter, gcc_counter, [author_gcc, author_total_counter]]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        process_master(comm)
    else:
        process_slave(comm)


if __name__ == '__main__':
    main()
