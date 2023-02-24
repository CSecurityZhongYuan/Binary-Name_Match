import os
import argparse
import site

import pandas as pd
import json
import random
import time
from itertools import combinations


class getDataset(object):
    def __init__(self,FLAG):
        self.inputdir = FLAG.input
        self.datacenterdir = FLAG.datacenter
        self.truepairs = None
        self.falsepairs = None
        self.file_list = self.scan_for_file(self.inputdir)
        self.min_node = int(FLAG.min_node)
        self.max_node = int(FLAG.max_node)

    def scan_for_file(self,start):
        file_list = []
        for root, dirs, files in os.walk(start):
            for file in files:
                file_list.append(os.path.join(start, file))
        print('Found ' + str(len(file_list)) + ' object files')
        random.shuffle(file_list)
        return  file_list

    def get_sim_pairs(self,file_a, file_b):
        pairs = list()
        softa = os.path.basename(file_a)
        softb = os.path.basename(file_b)
        with open(file_a, 'r') as afile:
            with open(file_b, 'r') as bfile:
                file_a_json = json.load(afile)
                a_functions = file_a_json['functions']
                file_b_json = json.load(bfile)
                b_functions = file_b_json['functions']

                for funa in a_functions:
                    if funa['file'] != 'UNKname':
                        a_name = funa['name']
                        a_file = funa['file']
                        cfga = a_file + a_name

                        for funb in b_functions:
                            if funb['file'] != 'UNKname':
                                b_name = funb['name']
                                b_file = funb['file']
                                cfgb = b_file + b_name
                                true_pair = dict()
                                if cfga == cfgb:
                                    lena =  len(funa['blocks'])
                                    lenb = len(funb['blocks'])
                                    if self.min_node<= lena <= self.max_node and self.min_node<= lenb <= self.max_node :
                                        true_pair['cfga'] = funa['blocks']
                                        true_pair['cfgb'] = funb['blocks']
                                        true_pair['softa'] = str(softa)
                                        true_pair['softb'] = str(softb)
                                        true_pair['namea'] = funa['name']
                                        true_pair['nameb'] = funb['name']
                                        true_pair['filea'] = funa['file']
                                        true_pair['fileb'] = funb['file']
                                        true_pair['addressa'] = funa['address']
                                        true_pair['addressb'] = funb['address']
                                        true_pair['similarity'] = '+1'
                                        pairs.append(true_pair)
                                        break
                                    else:
                                        pass

                                else:
                                    pass

                            else:
                                pass

        return pairs

    def get_dissim_pairs(self,file_a, file_b):
        pairs = list()
        softa =  os.path.basename(file_a)
        softb =  os.path.basename(file_b)
        with open(file_a, 'r') as afile:
            with open(file_b, 'r') as bfile:
                file_b_json = json.load(bfile)
                b_functions = file_b_json['functions']

                file_a_json = json.load(afile)
                a_functions = file_a_json['functions']

                for funa in a_functions:
                    a = random.choice(a_functions)
                    a_name = a['name']
                    a_file = a['file']
                    cfga = a_file + a_name
                    if a_file != 'UNKname':
                        for funb in b_functions:
                            b = random.choice(b_functions)
                            b_name = b['name']
                            b_file = b['file']
                            cfgb = b_file + b_name
                            false_pair = dict()
                            if b_file != 'UNKname':
                                if cfga != cfgb:
                                    lena = len(funa['blocks'])
                                    lenb = len(funb['blocks'])
                                    if self.min_node<= lena <= self.max_node and self.min_node<= lenb <= self.max_node:
                                        false_pair['cfga'] = a['blocks']
                                        false_pair['cfgb'] = b['blocks']
                                        false_pair['softa'] = str(softa)
                                        false_pair['softb'] = str(softb)
                                        false_pair['namea'] = a['name']
                                        false_pair['nameb'] = b['name']
                                        false_pair['filea'] = a['file']
                                        false_pair['fileb'] = b['file']
                                        false_pair['addressa'] = a['address']
                                        false_pair['addressb'] = b['address']
                                        false_pair['similarity'] = '-1'
                                        pairs.append(false_pair)
                                        break
                                    else:
                                        pass

                                else:
                                    pass

        return pairs

    def get_dataset(self):
        datacenterdir = self.datacenterdir
        file_list = self.file_list
        comp_files = list(combinations(file_list, 2))
        print(len(comp_files))

        sim_pairs = list()
        dis_pairs = list()
        for dataset in comp_files:
            file_a = dataset[0]
            file_b = dataset[1]
            # dataset similarity condition
            # dap-3.10_gcc-8.2.0_mips_32_O2_dappp.elf.json

            file_a_name = file_a.split('_')[-1].split('.')[0]
            file_b_name = file_b.split('_')[-1].split('.')[0]

            if file_a_name != file_b_name:
                pass
            else:
                t_pairs = self.get_sim_pairs(file_a, file_b)
                lent = len(t_pairs)
                if lent == 0:
                    pass
                else:
                    f_pairs = self.get_dissim_pairs(file_a, file_b)
                    lenf = len(f_pairs)
                    min = lent
                    if lenf < min:
                        min = lenf

                    print('t_pairs=', lent, 'f_pairs=', lenf, 'min=', min)

                    sim_pairs.extend(t_pairs[0:min])
                    dis_pairs.extend(f_pairs[0:min])
                    print('-----------------')

        true_data = os.path.join(datacenterdir, "dataset.true.json")
        random.shuffle(sim_pairs)

        with open(true_data, 'w') as simfile:
            json.dump(sim_pairs, simfile, indent=4, ensure_ascii=False)
        print(true_data + ':' + 'done')

        false_data = os.path.join(datacenterdir, "dataset.false.json")
        random.shuffle(dis_pairs)

        with open(false_data, 'w') as dissimfile:
            json.dump(dis_pairs, dissimfile, indent=4, ensure_ascii=False)
        print(false_data + ':' + 'done')

        self.truepairs = true_data
        self.falsepairs = false_data


    def creat_pairs(self,ratio):
        datacenterdir = self.datacenterdir
        random.seed(1234)


        ratios = ratio.split(':')

        trainset = dict()
        validset = dict()
        testset = dict()
        trainset['name'] = 'train dataset'
        validset['name'] = 'valid dataset'
        testset['name'] = 'test dataset'

        true_file = os.path.join(datacenterdir, "dataset.true.json")
        false_file = os.path.join(datacenterdir, "dataset.false.json")

        train_file = os.path.join(datacenterdir, "train_dataset.json")
        valid_file = os.path.join(datacenterdir, "valid_dataset.json")
        test_file = os.path.join(datacenterdir, "test_dataset.json")

        with open(true_file, 'r') as truefile:
            with open(false_file, 'r') as falsefile:
                true_dataset = json.load(truefile)
                true_len = len(list(true_dataset))

                false_dataset = json.load(falsefile)
                false_len = len(list(false_dataset))

                print(true_len, false_len, false_len+true_len)

                train_len = int( true_len * int(ratios[0])/10 + 1)
                valid_len = int( true_len * int(ratios[1])/10 + 1)
                test_len = int( true_len * int(ratios[2])/10)

                trainset['data'] = list()
                t_set = []
                t_set.extend(true_dataset[0:train_len])
                t_set.extend(false_dataset[0:train_len])
                random.shuffle(t_set)
                trainset['data'].append(t_set)

                validset['data'] = list()
                t_set = []
                t_set.extend(true_dataset[train_len:train_len+valid_len])
                t_set.extend(false_dataset[train_len:train_len+valid_len])
                random.shuffle(t_set)
                validset['data'].append(t_set)

                testset['data'] = list()
                t_set = []
                t_set.extend(true_dataset[train_len+valid_len:-1])
                t_set.extend(false_dataset[train_len+valid_len:-1])
                random.shuffle(t_set)
                testset['data'].append(t_set)

                print(len(trainset['data']),len(validset['data']),len(testset['data']))

                with open(train_file, 'w') as f:
                    json.dump(trainset, f, indent=4, ensure_ascii=False)
                print(train_file + ':' + 'done')

                with open(valid_file, 'w') as f:
                    json.dump(validset, f, indent=4, ensure_ascii=False)
                print(valid_file + ':' + 'done')

                with open(test_file, 'w') as f:
                    json.dump(testset, f, indent=4, ensure_ascii=False)
                print(test_file + ':' + 'done')

        return 0

    def dotest(self):
        test_file = os.path.join(self.datacenterdir, "test_dataset.json")
        with open(test_file, 'r') as f:
            test_dataset = json.load(f)
            print(test_dataset['name'])
            t_set = test_dataset['data'][0]
            index = 0
            for ind in t_set:
                print(ind['similarity'])
                cfga = ind['cfga']
                cfgb = ind['cfgb']

                for node in cfga:
                    if "M_" in node['normasms']:
                        print(node['normasms'])
                        index = index +1
                        print(ind['filea'])
                        print(ind['namea'])
                        print(ind['fileb'])
                        print(ind['nameb'])
                for node in cfgb:
                    if "M_" in node['normasms']:
                        print(node['normasms'])
                        index = index +1
                        print(ind['filea'])
                        print(ind['namea'])
                        print(ind['fileb'])
                        print(ind['nameb'])
                if index == 10:
                    exit(0)

            # true_len = len(t_set)
            # l = int(true_len/250)
            # print(l)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input ")
    parser.add_argument("-d", "--datacenter", type=str, help="output")
    parser.add_argument("-min", "--min_node", type=int, help="min_node")
    parser.add_argument("-max", "--max_node", type=int, help="max_node")
    args = parser.parse_args()

    # # min max node nums
    # min_node = 1
    # max_node = 1000
    # train:valid:test
    ratio = '8:1:1'
    pair_dataset = getDataset(args)

    pair_dataset.get_dataset()
    pair_dataset.creat_pairs(ratio)



