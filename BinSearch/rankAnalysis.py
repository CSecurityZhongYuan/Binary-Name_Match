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


def get_sim(vec_a, vec_b):
    vector_a = np.array(vec_a)
    vector_b = np.array(vec_b)
    num = float(np.dot(vector_a, vector_b))
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def find_index(args):
    dataname = args.database
    # dataname = './test_vec.json'
    database = read_dataset(dataname)
    search_base = list(database)
    index = 0
    for a_func in search_base:

        if a_func['fun_name'] == args.fun_name:
            # print(a_func['fun_vec'])
            print(a_func['fun_name'])
            print(a_func['file_name'])
            print(index)
        index = index + 1


def get_rank_index(args):
    ref_name = args.ref_name
    fun_database = args.database
    data_center = os.path.dirname(fun_database)
    model_name = args.model_name
    topN = int(args.top_n)
    # fun_database = './test_search_result.json'
    database = read_dataset(fun_database)
    # search_base = list(database)
    # print(len(search_base))
    # print(database)
    # a1= sorted(database,key=lambda x:x['sim'],reverse=True)
    # print(a1)


    # basename = pd.DataFrame(database,columns=['index','fun_name','file_name','sim'])
    basename = pd.DataFrame(database)

    basename= basename.sort_values(by='sim',ascending=False)

    fun_database = model_name+'_'+ref_name+'_top'+str(topN)+'.txt'

    fun_database = os.path.join(data_center, fun_database)

    np.savetxt(fun_database,basename[:topN],fmt='%s')

    print(fun_database + ':' + 'done')

def get_recall(args):
    ref_name = args.ref_name
    fun_database = args.database
    data_center = os.path.dirname(fun_database)
    model_name = args.model_name
    topN = int(args.top_n)
    fun_database = model_name + '_' + ref_name + '_top' + str(topN) + '.txt'
    fun_database = os.path.join(data_center, fun_database)
    Y1 = list()
    a_total = 12
    with open(fun_database,'r') as df:
        line =  df.readline()
        recall_name = line.split()[0]
        print("refername:",recall_name)
        recall = 1
        Y1.append(recall / a_total)
        for i in range(1,topN):
            line = df.readline()
            ref_name = line.split()[0]
            if ref_name == recall_name:
                recall = recall +1
            Y1.append(recall/a_total)

    fun_database = 'X2V_graph_with_kb_CVE-2022-0778_top200.txt'
    b_total = 10
    fun_database = os.path.join(data_center, fun_database)
    Y2 = list()
    with open(fun_database, 'r') as df:
        line = df.readline()
        recall_name = line.split()[0]
        print("refername:", recall_name)
        recall = 1
        Y2.append(recall / b_total)
        for i in range(1, topN):
            line = df.readline()
            ref_name = line.split()[0]
            if ref_name == recall_name:
                recall = recall + 1

            Y2.append(recall / b_total)

    step = 10

    D1 = list()
    D2 = list()
    for i in range(0,topN):
        if(i%step==0):
            D1.append(Y1[i])
    # D1.append(Y1[topN-1])

    for i in range(0,topN):
        if(i%step==0):
            D2.append(Y2[i])
    # D2.append(Y2[topN-1])


    print(len(Y2))
    print(len(D2))
    # exit(0)
    X = np.linspace(1,200,20)

    plt.figure(figsize=(8,6))
    font1 = {'family':'Times New Roman','weight':'normal','size':30}
    plt.xlabel('Top K',font1)
    plt.ylabel('The positive rate',font1)
    plt.plot(X,D1,color='b', lw=3,label="A recall" , linestyle='-.',marker='>')
    plt.plot(X, D2,color='g', lw=5,label="A recall", linestyle='--',marker='D')
    plt.savefig('./recall_rate.jpg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--database", type=str, help="database for binary search test ")
    parser.add_argument("-m", "--model_name", type=str, help="model name for binary search test ")
    parser.add_argument("-t", "--top_n", type=str, help="topN of similar function for binary search test ")
    parser.add_argument("-r", "--ref_name", type=str, help="function alise name for binary search test ")
    parser.add_argument("-f", "--fun_id", type=str, help="function  id for binary search test ")

    args = parser.parse_args()

    print(time.asctime(time.localtime(time.time())))

    # find_index(args)
    get_rank_index(args)

    # get_recall(args)



    print(time.asctime(time.localtime(time.time())))
    # trainer.scan_data()
    # trainer.test()
    # trainer.static_data()