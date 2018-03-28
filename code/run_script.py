import os
root = '/scratch0/swamiviv/GTA_Segnet/data/'
cmd = 'python train.py --dataroot ' + root + ' --gpu 0 --method LSD'
os.system(cmd)
