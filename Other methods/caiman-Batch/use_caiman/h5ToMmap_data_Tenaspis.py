# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:59:14 2019

@author:Somayyeh Soltanian-Zadeh
"""
import glob
import sys
sys.path.insert(0, 'C:\\Other methods\\CaImAn')
import caiman as cm

# simulation = sys.argv[1]
# datadir = 'D:\\ABO\\20 percent\\{}\\'.format(simulation)
# datadir = 'F:\\CaImAn data\\WEBSITE\\divided_data\\J123\\'
# datadir = 'E:\\ABO 175\\20 percent\\'
datadir = 'D:\\data_TENASPIS\\added_refined_masks\\'
AllFiles = glob.glob(datadir+"*.h5")


# LOAD MOVIE AND MEMORYMAP
for DataFile in AllFiles:
    fname_new = cm.save_memmap([DataFile],
                                base_name =DataFile.rsplit('\\', 1)[-1][0:-3]+'_memmap_', 
                                order='C')