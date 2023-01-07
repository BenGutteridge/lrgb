# a script to serve as a progress bar when running a bash script

import os
import os.path as osp
import sys
import time

"""
we want to put this between cfg runs on a bash script

takes the name of the run (which contains model and dataset info)

literally just has to add the name of it to a text doc
"""
 
# total arguments
args = sys.argv

filename = 'results/progress.txt'

try: 
  run_name = args[1]
  timestr = time.strftime("%d/%m, %H:%M:%S")
  message = ['\n{:30s} COMPLETED AT {:s}'.format(run_name, timestr)]
  with open(filename, 'a') as f:
    f.write('\n'.join(message))
except:
  if osp.exists(filename):
    os.remove(filename)
  with open(filename, 'w') as f:
    f.write('PROGRESS')