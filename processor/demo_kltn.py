#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

from data_gen.ntu_gendata import read_data_and_pre_norm
from data_gen.v_skeleton import draw_skeleton

class Demo(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):
        
        # initiate
        label_name_path = './resource/NTU-RGB-D/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name
        
        # read skeleton data
        skeleton, skeleton_original = read_data_and_pre_norm(self.arg.skeleton)
        data = torch.from_numpy(skeleton)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()
        
        
        # extract feature
        print('\nLan truyen qua kien truc...')
        self.model.eval()
        output, feature = self.model.extract_feature(data[0])
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print('Ket qua nhan dang: {}'.format(self.label_name[label]))
        print('Hoan thanh.')

        #visualize skeleton
        print("truc quan khung xuong goc")
        draw_skeleton('skeleton_original.avi', skeleton_original.transpose(3,1,2,0)[0]) # C,T,V,M
        print('Hoan thanh.')
        print('Luu ket qua')
        draw_skeleton('result.avi', skeleton_original.transpose(3,1,2,0)[0],label=self.label_name[label])
        print('Hoan thanh')
        

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--skeleton',
            default='',
            help='Path to video')
        # parser.add_argument('--openpose',
        #     default='3dparty/openpose/build',
        #     help='Path to openpose')
        # parser.add_argument('--output_dir',
        #     default='./data/demo_result',
        #     help='Path to save results')
        # parser.add_argument('--height',
        #     default=1080,
        #     type=int)
        parser.set_defaults(config='./config/st_gcn/ntu-xview/demo_old.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
