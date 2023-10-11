# python packages and directories
import tensorflow as tf
import numpy as np
import Func_Train
import pandas as pd
import matplotlib.pyplot as pl
import os
import xarray as xr

#homedir="/home/yousefi" #if using local
homedir=""              #if using hpc
#define all the necessary directories:
TRAIN_FILES=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/TRAIN_FILES"
DUMP_PLOT=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/DUMP_PLOT"
DUMP_RESULTS=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/DUMP_TRAIN_RESULTS"
PREDICT_FILES=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/PREDICT_FILES"
ATMOS_DATA=homedir+"/p/project/deepacf/kiste/patakchiyousefi1/DATASET/10-ATMOSCORRECT"
PPROJECT_DIR=homedir+"/p/project/deepacf/kiste/patakchiyousefi1"
PRODUCE_FILES=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/PRODUCTION_FILES"
STATS=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/STATS/"
STATSBASIC=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/STATSBASIC/"
WEIGHTS=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/WEIGHTS"
PARFLOWCLM=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM"

#HSAF_HRES directories:
HSAF_RET=homedir+"/p/largedata/slts/shared_data/obs_H-SAF_P-EUMETSAT/o.data/h61"
HSAF_OR=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/H_SAF"
HSAF_UTI=homedir+"/p/project/deepacf/kiste/patakchiyousefi1/IO/hsaf_ut/ftphsaf.meteoam.it/utilities/matlab_code"
HSAF_PP="/p/scratch/deepacf/kiste/patakchiyousefi1/H_SAF_PP"
HSAF_DUMP="/p/scratch/deepacf/kiste/patakchiyousefi1/H_SAF_DUMP"
HSAF_RG=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/HSAF_RG"
HRES_OR=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/H_RES"
HRES_PP=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/H_RES_PP"
HRES_DUMP=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/DUMP"
HSAF_PLOT=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/HSAF-PLOTS"
HRES_PLOT=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/HRES-PLOTS"
HRES_POST=homedir+"/p/scratch/deepacf/kiste/patakchiyousefi1/HRES_POST"