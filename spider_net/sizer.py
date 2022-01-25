import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

import numpy as np
import pickle as pkl
import pprint
import torch
import torch.nn as nn

from spider_net.data_loaders import load_data
from spider_net.helpers import mem_stats, clean, sizeof_fmt
from spider_net.net import Net
from spider_net.ops import commons
from spider_net.pruner import PrunableOperation
from spider_net.trainers import size_test



if __name__ == '__main__':
    if len(sys.argv)> 1 and sys.argv[1]=='o':
        # get sizes of individual operations in network
        # read args
        clean(verbose=False)
        C = int(sys.argv[3])
        dim = [int(x) for x in sys.argv[2:-1]]
        stride = int(sys.argv[-1])

        # build op
        op_mems = {}
        input_tensor = torch.zeros(dim,requires_grad=True).cuda()
        trials = 5
        criterion =  nn.CrossEntropyLoss()
        comparison = torch.tensor([0], dtype=torch.long).cuda()
        for op, f in commons.items():
            sizes = []
            
            for i in range(trials):
                sm = mem_stats(False)
                sms = []
                op_f = PrunableOperation(f, op, mem_size=0, c_in=C, stride=stride).cuda()  
                for _ in range(trials):
                    sms.append(mem_stats(False)-sm)
                    out = op_f(input_tensor)
                    sms.append(mem_stats(False)-sm)
                    out1 = out + out
                    sms.append(mem_stats(False)-sm)
                    loss = criterion(out.mean().reshape(1,1), comparison)
                    sms.append(mem_stats(False)-sm)
                    loss.backward()
                sizes.append(max(sms))
            clean(verbose=False)
            op_mems[op] = np.mean(sizes) / 1024 / 1024
        pp = pprint.PrettyPrinter(indent=0)
        pp.pprint(op_mems)
    else:
        # get size of entire model
        model = torch.load('pickles/sp_size_test.pt')
        out = size_test(model, verbose=False)
        with open("pickles/size_test_out.pkl", "wb") as f:
            pkl.dump(out, f)

