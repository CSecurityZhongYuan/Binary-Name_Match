import os
import json
import r2pipe
import networkx as nx
import matplotlib.pyplot as plt



class GetBinCode(object):
    def __init__(self,filename):
        self.filename = filename
        self.r2 = r2pipe.open(self.filename, flags=['-2'])
        self.function_list = []
        self.function_cfg = {}
        self.archs = "None"
        self.bits = "None"
        self.get_arch()
        self.result = None
        self.num = 0



    def get_arch(self):
        try:
            info = json.loads(self.r2.cmd('ij'))
            if 'bin' in info:
                self.archs = info['bin']['arch']
                self.bits = info['bin']['bits']
        except:
            print("Error loading file")
            self.archs = "None"
            self.bits = "None"


    def close(self):
        self.r2.quit()

    def get_function_list(self):

        r2 = self.r2

        r2.cmd('aaa')

        try:

            function_list = json.loads(r2.cmd('aflj'))

        except:
            function_list = []

        self.function_list = function_list

        return function_list

    def filter_reg(self,op):
        if "value" not in op:
            if "x86" == self.archs:
                op["value"] = "eax"
            if "arm" == self.archs:
                op["value"] = "zero"
            if "mips" == self.archs:
                op["value"] = "x0"

        return op["value"]

    def filter_imm(self,op):
        imm = int(op["value"])
        if -int(5000) <= imm <= int(5000):
            ret = str(hex(op["value"]))
        else:
            ret = str('HIMM')
        return ret


    def filter_mem(self,op):
        if "base" not in op:
            op["base"] = 0

        if op["base"] == 0:
            r = "[" + "MEM" + "]"
        else:
            reg_base = str(op["base"])
            disp = str(op["disp"])
            if "mips" == self.archs:
                r = '[' + reg_base + "+" + disp + ']'
            else:
                try:
                    scale = str(op["scale"])
                    r = '[' + reg_base + "*" + scale + "+" + disp + ']'
                except:
                    scale = 1
                    r = '[' + reg_base + "*" + str(scale) + "+" + disp + ']'
        return r

    def inst_normal(self,i):
        inst = "" + i["mnemonic"]

        for op in i["opex"]["operands"]:
            if op["type"] == 'reg':
                inst += " " + self.filter_reg(op)
            elif op["type"] == 'imm':
                inst += " " + self.filter_imm(op)
            elif op["type"] == 'mem':
                inst += " " + self.filter_mem(op)
            if len(i["opex"]["operands"]) > 1:
                inst = inst + ","

        if "," in inst:
            inst = inst[:-1]
        inst = inst.replace(" ", "_")

        return str(inst)

    def process_instructions(self,instructions,block):
        filtered_instructions = []
        for insn in instructions:
            # operands = []
            if 'opex' not in insn:
                continue
            stringized = self.inst_normal(insn)

            if "x86" == self.archs:
                stringized = "X_" + stringized
            elif "arm" == self.archs:
                stringized = "A_" + stringized
            elif "mips" == self.archs:
                stringized = "M_" + stringized
            else:
                stringized = "UNK_" + stringized
            filtered_instructions.append(stringized)
        api = ''
        for instruction in instructions:
            if "x86" == self.archs:
                if instruction['mnemonic'] == 'call':
                    parts = instruction['pseudo'].split(".")
                    api = parts[-1].split(" ")[0] + ' ' + api
            if "arm" == self.archs:
                if instruction['mnemonic'] == 'bl':
                    parts = instruction['pseudo'].split(".")
                    api = parts[-1].split(" ")[0] + ' ' + api
            if "mips" == self.archs:
                if instruction['mnemonic'] == 'bal':
                    parts = instruction['pseudo'].split(".")
                    api = parts[-1].split(" ")[0] + ' ' + api

        return filtered_instructions, api

    def process_block(self, block):
        r2 = self.r2

        disasm = []


        for op in block['ops']:
            if 'disasm' in op:
                disasm.append(op['disasm'])


        r2.cmd("s " + str(block['offset']))
        instructions = json.loads(r2.cmd("aoj " + str(len(block['ops']))))
        filtered_instructions, apis = self.process_instructions(instructions,block)
        return disasm, apis, filtered_instructions

    def function2cfg(self,func):

        r2 = self.r2
        r2.cmd('s ' + str(func["offset"]))
        self.num = self.num+1
        try:
            cfg = json.loads(r2.cmd('agfj ' + str(func["offset"])))
            # print(self.filename,self.num,'function done*')
        except:
            print('exception')
            cfg = []

        my_cfg = nx.DiGraph()

        if len(cfg) == 0:
            return my_cfg
        else:
            cfg = cfg[0]

        # add node information
        for block in cfg['blocks']:
            disasm, apis, filtered_instructions= self.process_block(block)
            my_cfg.add_node(block['offset'], orgiasms=disasm, apis=apis,
                            normasms=filtered_instructions )

        # structure block for cfg
        # flag = 0
        for block in cfg['blocks']:

            jump = []
            if 'jump' in block:
                if block['jump'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'], block['jump'])
                    jump.append(block['jump'])
            if 'fail' in block:
                if block['fail'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'], block['fail'])
                    jump.append(block['fail'])
            my_cfg.add_node(block['offset'],jumps=jump)


        return my_cfg

    def get_cfg(self):
        function_list = self.get_function_list()
        cfg_result = {}
        for function in function_list:

            cfg = self.function2cfg(function)
            address = hex(function['offset'])
            name = function["name"].split(".")[-1]

            self.function_cfg[function['offset']] = {'address': address, 'cfg': cfg, "name": name}

        return self.function_cfg

    def get_all(self):


        result = self.get_cfg()
        key = dict()
        key['arch'] = self.archs
        key['bit'] = self.bits
        key['filename'] = os.path.basename(self.filename)
        key['functions'] =list()

        for func_offset in result:
            function = dict()
            key['functions'].append(function)
            function['address'] = result[func_offset]['address']
            function['name'] = result[func_offset]['name']

            function['blocks'] = list()
            funcfg = result[func_offset]['cfg']

            for node in funcfg.nodes:
                bblock = dict()

                content = funcfg.nodes[node]
                bblock['id'] = node

                bb = ';'
                bblock['orgiasms']=bb.join(content['orgiasms'])
                bb = ';'
                bblock['normasms'] = bb.join(content['normasms'])

                bblock['apis'] = content['apis']

                bblock['jumps'] = content['jumps']

                function['blocks'].append(bblock)

        self.result = key
        return  self.result


    def do(self):
        if 'None' == self.archs:
            return "None"
        else:
            return self.get_all()

    def show_cfg(self,cfg):
        labels = {}
        for node in cfg.nodes():
            labels[node] = node
        nx.draw(cfg, labels=labels)
        plt.show()



