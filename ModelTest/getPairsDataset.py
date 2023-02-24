import argparse
import json
import time
import numpy as np
import networkx as nx
from scipy import sparse
import random
# from BlockFeaturesExtractor import BlockFeaturesExtractor

class pairsDataset(object):

    def __init__(self,word2id,api2id,dataset,seq_len,max_instructions,max_nodes,
                 api_seq,node_kb_max,max_path,path_len):
        self.word2id = self.read_word2id(word2id)
        self.api2id = self.read_api2id(api2id)
        self.dataset = self.read_dataset(dataset)
        self.max_instructions = max_instructions
        self.max_nodes = max_nodes

        self.max_api = node_kb_max

        self.api_seq = api_seq
        self.max_seq = seq_len

        self.max_path = max_path
        self.path_len = path_len
        self.cfg_path = 500

        self.maxint = 300
        self.minint = 10
        self.total = 0
        self.indint = 0

        self.acfg_length = 7
        # self.acfg_length = 8

        # self.bfe = BlockFeaturesExtractor()


    def read_word2id(self,word2id):
        with open(word2id, 'r') as f:
            word2idjson = json.load(f)
            return word2idjson

    def read_api2id(self,api2id):
        with open(api2id, 'r') as f:
            api2idjson = json.load(f)
            return api2idjson

    def read_dataset(self,dataset):
        with open(dataset, 'r') as f:
            datasetjson = json.load(f)
            self.datasetname = datasetjson['name']
            return datasetjson['data'][0]


    def get_ids(self,instructions):
        insts = str(instructions).split(';')
        # print(len(insts))
        ids = []
        # For each instruction we add +1 to its ID because the first
        # element of the embedding matrix is zero
        for x in insts:
            if x in self.word2id:
                ids.append(self.word2id[x] + 1)
            elif 'X_' in x:
                # print(str(x) + " is not a known x86 instruction")
                ids.append(self.word2id['X_UNK'] + 1)

            elif 'A_' in x:
                # print(str(x) + " is not a known arm instruction")
                ids.append(self.word2id['A_UNK'] + 1)
            else:
                # print("There is a problem " + str(x) + " does not appear to be an asm or arm instruction")
                ids.append(self.word2id['X_UNK'] + 1)
        # print(ids)
        return ids

    def get_api_ids(self,apis):
        # 'Api*noapi'
        words = apis
        ids = []
        temp = ''
        if words == '':
            temp=''
        else:
            for word in words.split():
                st = ''
                for w in word:
                    if w.isupper():
                        w = w.swapcase()
                        st = st + '_' + w
                    else:
                        st = st + w

                if st.startswith('_'):
                    word = st[1:]
                else:
                    word = st

                apiword = word

                if '_' in apiword:
                    apiword = apiword.split('_')
                    for api in apiword:
                        if len(api) != 0:
                            temp = temp + 'Api*' + api + ' '
                else:
                    temp = temp + 'Api*' + word + ' '
        if temp=='':
            token = 'Api*noapi'
            ids.append(self.api2id[token] + 1)
        else:
            for x in temp.split():
                if x in self.api2id:
                    ids.append(self.api2id[x] + 1)
                else:
                    token = 'Api*noapi'
                    ids.append(self.api2id[token] + 1)

        return ids
    def node_normalize(self, node):
        f = np.asarray(node[0:self.max_instructions])
        length = f.shape[0]
        if f.shape[0] < self.max_instructions:
            f = np.pad(f, (0, self.max_instructions - f.shape[0]), mode='constant')
        return f, length
    def kb_normalize(self, api):
        f = np.asarray(api[0:self.max_api])
        length = f.shape[0]
        if f.shape[0] < self.max_api:
            f = np.pad(f, (0, self.max_api - f.shape[0]), mode='constant')
        return f, length
    def function_normalize(self,cfg):
        max_nodes = self.max_nodes
        funmatrix = []
        funadj =  sparse.csr_matrix([1, 1])
        funlength = []
        fun_api = []
        api_lenghts = []

        lenghts = []
        fun_matrix = []
        api_matrix = []
        apilengths= []
        fun_adj = nx.adjacency_matrix(cfg)

        nodes = cfg.nodes(data=True)
        try:
            for i, n in enumerate(nodes):
                # print(n[1]['length'])
                lenghts.append(n[1]['length'])
                fun_matrix.append(n[1]['normasms'])
                fun_api.append(n[1]['api'])
                api_lenghts.append(n[1]['api_length'])

            if len(fun_api) <= max_nodes:
                # padding
                pad_lenght = max_nodes - len(fun_api)
                api_matrix = np.pad(fun_api, [(0, pad_lenght), (0, 0)], mode='constant')
                apilengths = api_lenghts + [0] * (max_nodes - len(api_lenghts))

            else:
                api_matrix = fun_api[0:self.max_nodes]
                apilengths = api_lenghts[0:self.max_nodes]

            if len(fun_matrix) <= max_nodes:
                # padding
                pad_lenght = max_nodes - len(fun_matrix)
                funmatrix = np.pad(fun_matrix, [(0, pad_lenght), (0, 0)], mode='constant')

                # padding
                pad_lenght = max_nodes - fun_adj.shape[0]
                if len(fun_matrix)== 1 :
                    funadj = np.zeros((max_nodes,max_nodes))
                else:
                    funadj = np.pad(fun_adj.todense(), [(0, pad_lenght), (0, pad_lenght)], mode='constant')

                funlength = lenghts + [0] * (max_nodes - len(lenghts))

            else:
                funmatrix = fun_matrix[0:self.max_nodes]
                funlength = lenghts[0:self.max_nodes]
                fun_adj = fun_adj.todense()
                # print("cfg > 150")
                funadj = np.asarray(fun_adj[0:self.max_nodes, 0:self.max_nodes])
               # print(funadj.shape)

        except:
            pass


        return funadj,funmatrix,funlength,api_matrix, apilengths
    def acfg_function_normalize(self,cfg):
        # 150*8
        max_nodes = self.max_nodes
        funmatrix = []
        funadj =  sparse.csr_matrix([1, 1])
        funlength = []
        fun_api = []
        api_lenghts = []

        lenghts = []
        fun_matrix = []
        api_matrix = []
        apilengths= []
        fun_adj = nx.adjacency_matrix(cfg)

        nodes = cfg.nodes(data=True)
        try:
            for i, n in enumerate(nodes):
                # print(n[1]['length'])
                lenghts.append(n[1]['length'])
                fun_matrix.append(n[1]['normasms'])
                fun_api.append(n[1]['api'])
                api_lenghts.append(n[1]['api_length'])

            if len(fun_api) <= max_nodes:
                # padding
                pad_lenght = max_nodes - len(fun_api)
                api_matrix = np.pad(fun_api, [(0, pad_lenght), (0, 0)], mode='constant')
                apilengths = api_lenghts + [0] * (max_nodes - len(api_lenghts))

            else:
                api_matrix = fun_api[0:self.max_nodes]
                apilengths = api_lenghts[0:self.max_nodes]

            if len(fun_matrix) <= max_nodes:
                # padding
                pad_lenght = max_nodes - len(fun_matrix)
                funmatrix = np.pad(fun_matrix, [(0, pad_lenght), (0, 0)], mode='constant')

                # padding
                pad_lenght = max_nodes - fun_adj.shape[0]
                if len(fun_matrix)== 1 :
                    funadj = np.zeros((max_nodes,max_nodes))
                else:
                    funadj = np.pad(fun_adj.todense(), [(0, pad_lenght), (0, pad_lenght)], mode='constant')

                funlength = lenghts + [0] * (max_nodes - len(lenghts))

            else:
                funmatrix = fun_matrix[0:self.max_nodes]
                funlength = lenghts[0:self.max_nodes]
                fun_adj = fun_adj.todense()
                # print("cfg > 150")
                funadj = np.asarray(fun_adj[0:self.max_nodes, 0:self.max_nodes])
               # print(funadj.shape)

        except:
            pass

        return funadj,funmatrix,funlength,api_matrix, apilengths
    def get_data_from_acfg(self,cfg):
        my_cfg = nx.DiGraph()
        a_api_ids = []
        for node in cfg:
            id = node['id']
            ids = [node['features'][x] for x in node['features']]
            api = self.get_api_ids(node['apis'])
            a_api_ids.extend(api)
            # xref = node['xrefs']
            api,la = self.kb_normalize(api)
            my_cfg.add_node(id,length=len(ids),api=api, normasms=ids,api_length = la)
        for node in cfg:
            jump = len(node['jumps'])
            id = node['id']
            if jump == 0:
                pass
            else:
                for n in node['jumps']:
                    my_cfg.add_edge(id, n)
        # print(len(my_cfg.nodes))
        a_api_ids, _ = self.get_api_normalize(a_api_ids)
        funadj, funmatrix, funlength, api_matrix, apilengths = self.acfg_function_normalize(my_cfg)
        return funadj, funmatrix, funlength, api_matrix, apilengths, a_api_ids
    def get_data_from_cfg(self,cfg):
        my_cfg = nx.DiGraph()
        a_api_ids = []
        for node in cfg:
            id = node['id']
            ids = self.get_ids(node['normasms'])

            api_ids = self.get_api_ids(node['apis'])
            a_api_ids.extend(api_ids)

            api = self.get_api_ids(node['apis'])
            # xref = node['xrefs']

            i,li = self.node_normalize(ids)
            api,la = self.kb_normalize(api)
            my_cfg.add_node(id,length=li,api=api, normasms=i,api_length = la)

        for node in cfg:
            jump = len(node['jumps'])
            id = node['id']
            if jump == 0:
                pass
            else:
                for n in node['jumps']:
                    my_cfg.add_edge(id, n)

        # print(len(my_cfg.nodes))
        a_api_ids, _ = self.get_api_normalize(a_api_ids)
        funadj,funmatrix,funlength,api_matrix, apilengths = self.function_normalize(my_cfg)
        return funadj,funmatrix,funlength,api_matrix, apilengths,a_api_ids
    def get_seq_from_cfg(self,cfg):
        a_ids = []
        a_api_ids = []

        for node in cfg:
            id = node['id']
            ids = self.get_ids(node['normasms'])
            a_ids.extend(ids)
            api_ids = self.get_api_ids(node['apis'])
            a_api_ids.extend(api_ids)

            self.indint = self.indint + 1

        g_len = len(a_ids)
        self.total = self.total +g_len
        if g_len > int(self.maxint):
            self.maxint = g_len
        else:
            if g_len < int(self.minint):
                self.minint = g_len
        a_ids,a_length = self.get_seq_normalize(a_ids)
        a_api_ids,a_api_length = self.get_api_normalize(a_api_ids)

        return a_ids,a_api_ids,a_length,a_api_length

    def get_edge_from_cfg(self, cfg):
        # 100*150  seq*len
        my_cfg = nx.DiGraph()
        my_cfg.add_node('start')
        my_cfg.add_node('end')
        for node in cfg:
            my_cfg.add_node(node['id'], asms=node['normasms'], apis=node['apis'])

        for node in cfg:
            for jump_id in node['jumps']:
                my_cfg.add_edge(node['id'], jump_id)

        edges = []

        for one in my_cfg.edges():
            edges.append(one[0])
            edges.append(one[1])
        # print(len(edges))
        asms = []

        for node in edges:
            ids = self.get_ids(my_cfg.nodes[node]['asms'])
            asms.extend(ids)
        # print(asms)
        a_ids, length = self.get_seq_normalize(asms)
        return a_ids

    def get_path_from_cfg(self,cfg):
        # 100*150  seq*len
        my_cfg = nx.DiGraph()
        my_cfg.add_node('start')
        my_cfg.add_node('end')
        for node in cfg:
            my_cfg.add_node(node['id'], asms=node['normasms'], apis=node['apis'])

        my_cfg.add_edge('start', cfg[0]['id'])

        for node in cfg:
            if len(node['jumps']) == 0:
                my_cfg.add_edge(node['id'], 'end')
            else:
                for jump_id in node['jumps']:
                    my_cfg.add_edge(node['id'], jump_id)

        # try:
        #     paths = nx.all_simple_paths(my_cfg, source='start', target='end')
        # except:
        #     pass
        #
        # allpath = []
        #
        # for path in paths:
        #     path.pop()
        #     del (path[0])
        #     allpath.append(path)


        walk_length = len(my_cfg.nodes())
        paths = []
        for _ in range(self.max_path):
            path = self.cfg_random_walk(my_cfg, source='start', walk_length=walk_length)
            paths.append(path)

        allpath = []
        for path in paths:
            path.pop()
            del (path[0])
            allpath.append(path)


        a_ids = []
        seqpaths = allpath
        if len(seqpaths):
            for path in seqpaths:
                asm = []
                for node in path:
                    asm_ids = self.get_ids(my_cfg.nodes[node]['asms'])
                    asm.extend(asm_ids)
                asm, a_length = self.get_pathlen_normalize(asm)
                a_ids.append(asm)
        else:
            zero = [0 for x in range(self.path_len)]
            a_ids.append(zero)
        a_ids = self.path_normalize(a_ids)
        return a_ids

    def cfg_random_walk(self,my_cfg, source, walk_length):
        path = [source]
        while len(path) < walk_length:
            current_node = path[-1]
            current_node_neighors = list(my_cfg.neighbors(current_node))
            if len(current_node_neighors) > 0:
                path.append(random.choice(current_node_neighors))
            else:
                break

        return path
    def path_normalize(self,paths):
        pathmatrix = []
        if len(paths) <= self.max_path:
            # padding
            pad_lenght = self.max_path - len(paths)
            pathmatrix = np.pad(paths, [(0, pad_lenght), (0, 0)], mode='constant')
        else:
            pathmatrix = paths[0:self.max_path]

        return pathmatrix

    def get_maxmin(self):
        print('max ',self.maxint)
        print('min ',self.minint)
        print('mean',int(self.total/self.indint))
        print('ind',self.indint)
    def get_seq_normalize(self,ids):

        f = np.asarray(ids[0:self.max_seq])

        length = f.shape[0]
        if f.shape[0] < self.max_seq:
            f = np.pad(f, (0, self.max_seq - f.shape[0]), mode='constant')
        return f ,length


    def get_api_normalize(self,a_api_ids):
        f = np.asarray(a_api_ids[0:self.api_seq])
        length = f.shape[0]
        if f.shape[0] < self.api_seq:
            f = np.pad(f, (0, self.api_seq - f.shape[0]), mode='constant')
        return f,length

    def get_batch_pairs(self,batch_data):

        g1_adj = []
        g1_matrix = []
        g1_length = []
        g1_api_length = []
        g1_api = []
        lable = []
        g2_adj = []
        g2_matrix = []
        g2_length = []
        g2_api_length = []
        g2_api = []

        ga_api =[]
        gb_api = []
        # batch_size*150*150

        for ind in batch_data:
            lab = float(ind['similarity'])
            # lab = int(ind['similarity'])
            lable.append(lab)
            cfga = ind['cfga']
            cfgb = ind['cfgb']
            # funadj,funmatrix,funlength,fun_api_ids, fun_api_length
            a_adj, a_matrix, a_length,a_api,a_api_length,A_api = self.get_data_from_cfg(cfga)
            g1_adj.append(a_adj)
            g1_matrix.append(a_matrix)
            g1_length.append(a_length)
            g1_api.append(a_api)
            g1_api_length.append((a_api_length))

            b_adj, b_matrix, b_length,b_api,b_api_length,B_api= self.get_data_from_cfg(cfgb)

            # print('b_matrix', np.asarray(b_matrix).shape)

            g2_adj.append(b_adj)
            g2_matrix.append(b_matrix)
            g2_length.append(b_length)
            g2_api.append(b_api)
            g2_api_length.append((b_api_length))

            ga_api.append(A_api)
            gb_api.append(B_api)


        out = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,
               lable,g1_api, g2_api, g1_api_length, g2_api_length,ga_api,gb_api]
        return out

    def get_acfg_pairs(self,batch_data):

        g1_adj = []
        g1_matrix = []
        g1_length = []
        g1_api_length = []
        g1_api = []
        lable = []
        g2_adj = []
        g2_matrix = []
        g2_length = []
        g2_api_length = []
        g2_api = []

        ga_api = []
        gb_api = []
        # batch_size*150*150

        for ind in batch_data:
            lab = float(ind['similarity'])
            # lab = int(ind['similarity'])
            lable.append(lab)
            cfga = ind['cfga']
            cfgb = ind['cfgb']
            # funadj,funmatrix,funlength,fun_api_ids, fun_api_length
            a_adj, a_matrix, a_length,a_api,a_api_length,A_api= self.get_data_from_acfg(cfga)
            g1_adj.append(a_adj)
            g1_matrix.append(a_matrix)
            g1_length.append(a_length)
            g1_api.append(a_api)
            g1_api_length.append((a_api_length))

            b_adj, b_matrix, b_length,b_api,b_api_length,B_api= self.get_data_from_acfg(cfgb)

            # print('b_matrix', np.asarray(b_matrix).shape)

            g2_adj.append(b_adj)
            g2_matrix.append(b_matrix)
            g2_length.append(b_length)
            g2_api.append(b_api)
            g2_api_length.append((b_api_length))

            ga_api.append(A_api)
            gb_api.append(B_api)


        out = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,
               lable,g1_api, g2_api, g1_api_length, g2_api_length,ga_api,gb_api]
        return out
    def get_pairs(self,pair):

        g1_matrix = []
        g1_api = []
        g1_length = []
        g1_api_length = []
        lable = []
        g2_matrix = []
        g2_api = []
        g2_length = []
        g2_api_length = []
        # g1_acfg = []
        # g2_acfg = []

        # batch_size*seq
        for ind in pair:
            lab = float(ind['similarity'])
            lable.append(lab)
            cfga = ind['cfga']
            cfgb = ind['cfgb']

            a_matrix,a_api,a_length,a_api_length= self.get_seq_from_cfg(cfga)
            b_matrix,b_api,b_length,b_api_length= self.get_seq_from_cfg(cfgb)

            g1_matrix.append(a_matrix)
            g2_matrix.append(b_matrix)

            g1_api.append(a_api)
            g2_api.append(b_api)

            g1_length.append(a_length)
            g2_length.append(b_length)

            g1_api_length.append(a_api_length)
            g2_api_length.append(b_api_length)

            # g1_acfg.append(a_acfg)
            # g2_acfg.append(b_acfg)
        out = [g1_matrix, g2_matrix,lable,g1_api,g2_api,g1_length,g2_length,g1_api_length,g2_api_length]
        return out




