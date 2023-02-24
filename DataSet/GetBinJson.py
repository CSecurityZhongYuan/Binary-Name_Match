import datetime
from GetBinCode import GetBinCode
import argparse
import os
import random
import json
from multiprocessing import Pool
from tqdm import tqdm

def scan_for_file(start):
    file_list = []
    for root, dirs, files in os.walk(start):
        for file in files:
            # if 'gcc-8.2.0_x86_64' in file:
            #     file_list.append(os.path.join(root, file))
            # if 'gcc-8.2.0_arm_64' in file:
            #     file_list.append(os.path.join(root, file))
            # if 'gcc-8.2.0_mips_64' in file:
            #     file_list.append(os.path.join(root, file))
            file_list.append(os.path.join(root, file))

    return file_list

def dump_data_to_file(data, output_data_file):
    with open(output_data_file, "w") as f:
        for d in data:
            d = d.replace('\t', '')
            if isinstance(d, list):
                f.write(" ".join(d) + "\n")
            else:
                f.write(d + "\n")


def getbinarycode(filename,outdir):
    bc = GetBinCode(filename)
    print(filename + ':' + 'binary code beginning')
    result = bc.do()
    if result=='None':
        print(filename + ':' + 'binary unknown ')
        return 0
    else:
        filename = os.path.basename(filename)
        outbinarycode = os.path.join(outdir, filename + ".json")
        #
        with open(outbinarycode,'w') as outfile:
            json.dump(result,outfile,indent=4,ensure_ascii=False)
        return 0

def dowork(item):
    filename = item[0]
    outdir = item[1]
    print("Analyzing file: {}".format(filename))
    getbinarycode(filename,outdir)
    print("Analyzing file: {} done".format(filename))
    return 0

def getBinSet(inputfiledir,datacenter):

    inputfilelist = scan_for_file(inputfiledir)
    print('Found ' + str(len(inputfilelist)) + ' object files')

    random.shuffle(inputfilelist)
    fileargs = [(f, datacenter) for f in inputfilelist]

    p = Pool(processes=None, maxtasksperchild=10)
    for _ in tqdm(p.imap_unordered(dowork, fileargs), total=len(inputfilelist)):
        pass


    p.close()
    p.join()

def get_dataset_statistic(datacenter):
    inputfilelist = scan_for_file(datacenter)
    print('Found ' + str(len(inputfilelist)) + ' object files')
    random.shuffle(inputfilelist)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Executable file to analyze")
    parser.add_argument("-o", "--datacenter", type=str, help="data file ")
    args = parser.parse_args()

    inputfiledir = args.input
    datacenter =args.datacenter

    start_time = datetime.datetime.now()

    getBinSet(inputfiledir,datacenter)

    end_time = datetime.datetime.now()
    during_time = end_time - start_time
    print(during_time)


