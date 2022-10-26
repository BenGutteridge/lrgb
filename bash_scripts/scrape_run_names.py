#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:22:15 2022

@author: beng
"""
import os

os.chdir('/Users/beng/Documents/lrgb')
completed_runs = []
for dir in sorted(os.listdir('results/results_polished')):
    if len(dir)>10:
        completed_runs.append(dir)
            

for path, subdirs, files in sorted(os.walk('configs/')):
    for name in files:    
        file = os.path.join(path, name)
        if file.endswith('.yaml'):
            if not 'd=' in file:
                name = file.rsplit('/', 1)[-1][:-5] # get rid of yaml
                if name not in completed_runs:
                    if 'voc' in file:
                        print(file)
        