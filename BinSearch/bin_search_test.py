import os

import argparse


import random
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def read_dataset(dataset):
    with open(dataset, 'r') as f:
        datasetjson = json.load(f)
        return datasetjson
def get_sim(vec_a,vec_b):
    vector_a = np.array(vec_a)
    vector_b = np.array(vec_b)
    num = float(np.dot(vector_a,vector_b))
    denom = np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    cos = num/denom
    sim = 0.5+ 0.5*cos
    return  sim

def find_index(args):
    dataname= args.database
    # dataname = './test_vec.json'
    database = read_dataset(dataname)
    search_base = list(database)
    index= 0
    target = []
    for a_func in search_base:
        if a_func['fun_name'] == args.fun_name:
            # print(a_func['fun_vec'])
            # print(a_func['fun_name'],a_func['file_name'],index)
            target.append(index)
        index = index+1
    return target

def findone(data_center,search_base,source_index,match_num):

    result = list()
    cve_vec = search_base[source_index]['fun_vec']
    print("-----------------------")
    print("file_name", search_base[source_index]['file_name'])

    index = 0

    ratio = 0.15
    len_a = int(search_base[source_index]['fun_nodes'])
    for a_func in search_base:

        vec_a = cve_vec
        vec_b = a_func['fun_vec']
        len_b = int(search_base[index]['fun_nodes'])

        if int(len_b*(1-ratio))<=len_a<=int(len_b*((1+ratio))):

            sim = get_sim(vec_a, vec_b)

            # a_result = {'fun_name': a_func['fun_name'], 'file_name': a_func['file_name'], "sim": sim}
            a_result = {'fun_name': a_func['fun_name'], 'fun_nodes': a_func['fun_nodes'],
                    'file_name': a_func['file_name'], "sim": sim}



            result.append(a_result)
        index = index + 1
    result_len = len(result)
    print('match nums after filter:',result_len)
    fun_database = 'temp_search_result.json'
    fun_database = os.path.join(data_center, fun_database)

    with open(fun_database, 'w') as basefile:
        json.dump(result, basefile, indent=4, ensure_ascii=False)
    # print(fun_database + ':' + 'done')

    get_recall(args, fun_database,match_num,result_len)
    os.remove(fun_database)

def bin_search_test(args):
    dataname= args.database
    data_center = os.path.dirname(dataname)

    ref_funs = find_index(args)
    match_num = len(ref_funs)
    database = read_dataset(dataname)
    search_base = list(database)
    for source_index in ref_funs:
        print(source_index)
        findone(data_center,search_base,source_index,match_num)



def get_rank_index(fun_database,top_n,data_center):
    topN = top_n

    database = read_dataset(fun_database)

    basename = pd.DataFrame(database)

    basename= basename.sort_values(by='sim',ascending=False)

    top_result = 'top'+str(topN)+'.txt'

    result = os.path.join(data_center, top_result)

    np.savetxt(result,basename[:topN],fmt='%s')

    # print(result + ':' + 'done')

def get_recall(args,fun_database,match_num,result_len):

    data_center = os.path.dirname(fun_database)
    top_n = int(args.top_n)
    if top_n >= result_len:
        top_n = result_len
    get_rank_index(fun_database,top_n,data_center)

    top_result = 'top'+str(top_n)+'.txt'
    top_txt = os.path.join(data_center, top_result)

    a_total = match_num
    with open(top_txt,'r') as df:
        line =  df.readline()
        recall_name = line.split()[0]

        recall = 1

        for i in range(1,top_n):
            line = df.readline()
            ref_name = line.split()[0]
            # print(ref_name)
            if ref_name == recall_name:
                recall = recall +1

        print("refername:",recall_name,"number:",recall,"recall rate:",recall/a_total)

    os.remove(top_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--database", type=str, help="database for binary search test ")
    parser.add_argument("-n", "--fun_name", type=str, help="function name for binary search test ")
    # parser.add_argument("-m", "--model_name", type=str, help="model name for binary search test ")
    # parser.add_argument("-s", "--source_index", type=str, help="function index for binary search test ")
    parser.add_argument("-t", "--top_n", type=str, help="function alise name for binary search test ")

    args = parser.parse_args()

    print(time.asctime(time.localtime(time.time())))


    # find_index(args)
    bin_search_test(args)

    print(time.asctime(time.localtime(time.time())))
    # trainer.scan_data()
    # trainer.test()
    # trainer.static_data()