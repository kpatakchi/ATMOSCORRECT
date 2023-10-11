#homedir="/home/yousefi" #if using local
homedir=""              #if using hpc
#define all the necessary directories:
PPROJECT_DIR="/p/project/deepacf/kiste/patakchiyousefi1/"
PSCRATCH_DIR="/p/scratch/deepacf/kiste/patakchiyousefi1/"

# General directories
TRAIN_FILES=homedir+PSCRATCH_DIR+"TRAIN_FILES"
DUMP_PLOT=homedir+PSCRATCH_DIR+"DUMP_PLOT"
DUMP_RESULTS=homedir+PSCRATCH_DIR+"DUMP_TRAIN_RESULTS"
PREDICT_FILES=homedir+PSCRATCH_DIR+"PREDICT_FILES"
ATMOS_DATA=homedir+PPROJECT_DIR+"DATASET/10-ATMOSCORRECT"
PRODUCE_FILES=homedir+PSCRATCH_DIR+"PRODUCTION_FILES"
STATS=homedir+PSCRATCH_DIR+"STATS/"
STATSBASIC=homedir+PSCRATCH_DIR+"STATSBASIC/"
WEIGHTS=homedir+PSCRATCH_DIR+"WEIGHTS"

#HSAF_HRES directories:
HSAF_RET=homedir+"/p/largedata/slts/shared_data/obs_H-SAF_P-EUMETSAT/o.data/h61"
HSAF_OR=homedir+PSCRATCH_DIR+"H_SAF"
HSAF_UTI=homedir+"/p/project/deepacf/kiste/patakchiyousefi1/IO/hsaf_ut/ftphsaf.meteoam.it/utilities/matlab_code"
HSAF_PP=PSCRATCH_DIR+"H_SAF_PP"
HSAF_DUMP=PSCRATCH_DIR+"H_SAF_DUMP"
HSAF_RG=homedir+PSCRATCH_DIR+"HSAF_RG"
HRES_OR=homedir+PSCRATCH_DIR+"H_RES"
HRES_PP=homedir+PSCRATCH_DIR+"H_RES_PP"
HRES_DUMP=homedir+PSCRATCH_DIR+"DUMP"
HSAF_PLOT=homedir+PSCRATCH_DIR+"HSAF-PLOTS"
HRES_PLOT=homedir+PSCRATCH_DIR+"HRES-PLOTS"
HRES_POST=homedir+PSCRATCH_DIR+"HRES_POST"

# PARFLOW Directories
PARFLOWCLM=homedir+PSCRATCH_DIR+"PARFLOWCLM"
ORIG_HRES=PARFLOWCLM+"/sim/ADAPTER_DE05_ECMWF-HRES_detforecast__FZJ-IBG3-ParFlowCLM380D_v03bJuwelsGpuProdClimatologyTl_PRhourly/forcing/o.data.MARS_retrieval"
COR_HRES=PARFLOWCLM+"/sim/ADAPTER_DE05_ECMWF-HRES_detforecast__FZJ-IBG3-ParFlowCLM380D_v03bJuwelsGpuProdClimatologyTl_PRhourly_PRCORRECTED/forcing/o.data.MARS_retrieval"

# GRDC:
GRDC_DIR=homedir+PPROJECT_DIR+"DATASET/11-GRDC"