# ATMOSCORRECT & PARFLOWCLM

_Author: Kaveh Patakchi Yousefi_

[_k.patakchi.yousefi@fz-juelich.de_](mailto:k.patakchi.yousefi@fz-juelich.de)

JUNE 2023

# 1)Data Preparation

## 1.1)Preprocess HSAF data:

- Download HSAF data (refer to `HSAF_DOWNLOAD.docx`)
- Run `DATA_MANAGE.ipynb` 2.2 to copy downloaded data from `HSAF_RET` to `HSAF_OR `
- Run HSAF\_PP.ipynb 1 to preprocess HSAF data

## 1.2)Preprocess HRES data:

- Run `DATA_MANAGE.ipynb` 1.2 to unzip and copy data from `HRES_RET` to `HRES_OR`
- Run `HRES_PP.ipynb` to preprocess HRES data in `HRES_OR` and save it in `HRES_PP`

## 1.3)Move the Preprocessed Data to /p/project

cp /p/scratch/deepacf/kiste/patakchiyousefi1/HSAF\_RG/HSAF\_PP\_202007\_202304.remapbil.cr.nc /p/project/deepacf/kiste/patakchiyousefi1/DATASET/10-ATMOSCORRECT/HSAF\_pr.nc

cp /p/scratch/deepacf/kiste/patakchiyousefi1/H\_RES\_PP/HRES\_PP\_202007\_202304.nc /p/project/deepacf/kiste/patakchiyousefi1/DATASET/10-ATMOSCORRECT/HRES\_pr.nc

## 1.4)Preprocessing in Python (HSAF\_HRES\_PP2.ipynb)

- To fill up the missing hours in HSAF data with np.nan, run ` HSAF_HRES_PP2.ipynb` (will save as nc.nc, must manually delete the old file and rename to nc). This code will also replace HRES's coordinates with HSAF's to avoid misalignment.

# 2)Training and Production

# 3)Calculate Statistics

- Run `STATS.ipynb` to calculate the statistics (e.g., errors etc).

# 4)Visualize/Tabulate Statistics

- Run `VISTAB.ipynb` to visualize or tabulate the statistics calculated in `STATS.ipynb` or other notebooks with statistical outputs.

# 5)HRES Data Post-processing

- Run `HRES_POSTP` to post-process HRES\_pr.nc to several .nc files in `FORCING` folder.

# 6)Run ParFlowCLM

SCRATCHDIR=/p/scratch/deepacf/kiste/patakchiyousefi1

Copy the climatology gitlab from Alexandre's folder to my scratch:

cp -r /p/scratch/pfgpude05/belleflamme1/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly/ctrl /p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM/

Copy the geo folder containing the static fields:
 cp -r /p/scratch/pfgpude05/belleflamme1/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly/geo /p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM/

**Adapt the main path in ctrl/ParFlowCLMSetup.ksh (l. 17) and in ctrl/ParFlowCLM\_ini.txt (l.1)**

Changes made in ParFlowCLMSetup.ksh

- Change DIRBASE
- Add # 2023-04-07 k.patakchi.yousefi@fz-juelich.de patakchiyousefi1

Changes made in ParFlowCLM\_ini.txt:

- Change DIRBASE
- Comment out DIRLARGE in line 14 (for now)

All v03 scripts are already prepared to run on JUWELS Booster and are based on Stages/2023. You might only need to change "ACCOUNT2=$YourComputeTimeProject" in ctrl/ParFlowCLMStarterJuwelsChainJob\_RunSim\_2.ksh and maybe in ctrl/ParFlowCLMStarterJuwelsChainJob\_PrePro\_2.ksh

To run your simulations without overland flow routing (runs much faster). So, you need to change the branch in ctrl/ParFlowCLMSetup.ksh, l. 52 to "cuda11\_mgsemi\_jacobian\_fix\_no\_routing\_no\_overlandflow\_opt\_clmrst"

Go to scratch. mkdir sim, mkdir ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly and fix the folders according to Alexandre's folders.

**To compile ParFlow, login on JUWELS Booster, go to the ctrl folder and simply do "./ParFlowCLMSetup.ksh" on a front node (has to be JUWELS Booster !)**

The HRES forcing files have to be moved (and untared, if needed) to forcing/o.data.MARS\_retrieval.

We added the original HRES data into this folder: "p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly/forcing/o.data.MARS\_retrieval/"

Must adapt ctrl/ParFlowCLMPreProAtmForcing\_2.ksh to your forcing files. The following changes were made:

- added # 2023-04-07 k.patakchi.yousefi@fz-juelich.de patakchiyousefi
- Commented out line 167 #cp ${ECMWF\_DATASTREAM}/${infile} ${DIRFORCINGIN}

The initialization files are located here (you will start from climatology v02): /p/largedata/dpfgpude05/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v02bJurecaGpuProdClimatologyTl\_PRhourly/simres

If you start, for example, on 2020-01-01T12:00Z, you need
 (i) 2019/ParFlowCLM\_DE05.out.2019123112.00024.nc which is the 3D pressure from ParFlow at 2020-01-01T12:00Z, which is used to restart ParFlow
 (ii) 2019/clm.rst.2019123112.00024.tar which are the restart files for CLM

In this case, I would need the following files:

Initfile\_parflow="/p/largedata/dpfgpude05/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v02bJurecaGpuProdClimatologyTl\_PRhourly/simres/2020/ParFlowCLM\_DE05.out.2020063012.00024.nc"

Initfile\_clm="/p/largedata/dpfgpude05/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v02bJurecaGpuProdClimatologyTl\_PRhourly/simres/2020/clm.rst.2020063012.00024.tar"

Make the directories /simres/2020 etc and copy these two files:

cp /p/largedata/dpfgpude05/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v02bJurecaGpuProdClimatologyTl\_PRhourly/simres/2020/ParFlowCLM\_DE05.out.2020063012.00024.nc $SCRATCHDIR/PARFLOWCLM/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly/simres/2020/

cp /p/largedata/dpfgpude05/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v02bJurecaGpuProdClimatologyTl\_PRhourly/simres/2020/clm.rst.2020063012.00024.tar $SCRATCHDIR/PARFLOWCLM/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly/simres/2020/

To do the simulations, use the \*\_2.ksh scripts. With these scripts, you will have two jobs per month (01-15, 16-end) to ensure to stick within the 24h run time limit

**For the preprocessing:
 login on **** JUWELS ****, go to ctrl: ./ParFlowCLMStarterJuwelsChainJob\_PrePro\_2.ksh -\> indicate the start date and the number of months in the script**

Changes in this file:

- EMAILUSER
- NO\_OF\_MONTHS
- INIDATE
- # 2023-04-07 k.patakchi.yousefi@fz-juelich.de patakchiyousefi1 adapted from goergen1

**For the simulations:
 login on **** JUWELS Booster ****, go to ctrl: ./ParFlowCLMStarterJuwelsChainJob\_RunSim\_2.ksh -\> indicate the start date and the number of months in the script. You can also indicate the job ID of a queuing job to extend the job chain without interruption**

Changes in this file:

- EMAILUSER
- NO\_OF\_MONTHS
- INIDATE
- # 2023-04-07 k.patakchi.yousefi@fz-juelich.de patakchiyousefi1 adapeted from goergen1

Now, for ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly\_HRES\_CORRECTED:

Firstly, choose the right prediction file (e.g., PREDICT\_FILE="HRES\_C\_predict\_data.mse.32.0.01.0.0001.0.25.2.2.8.0.1.64\_mse.32.0.01.0.0001.0.25.2.2.8.0.1.64.nc") and run the second part of `HRES_POSTP.ipynb`. This will generate the corrected HRES data in the original format in the forcing file.

**Now, copy all the necessary files from ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly:**

CORR\_RUN\_PATH=/p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly\_HRES\_CORRECTED

ORIG\_RUN\_PATH=/p/scratch/deepacf/kiste/patakchiyousefi1/PARFLOWCLM/sim/ADAPTER\_DE05\_ECMWF-HRES\_detforecast\_\_FZJ-IBG3-ParFlowCLM380D\_v03bJuwelsGpuProdClimatologyTl\_PRhourly

cp -r $ORIG\_RUN\_PATH/geo $CORR\_RUN\_PATH/

cp -r $ORIG\_RUN\_PATH/simres $CORR\_RUN\_PATH/

cp -r $ORIG\_RUN\_PATH/ctrl $CORR\_RUN\_PATH/

**Now, we have to modify all the (necessary) files in this new directory's ctrl:**

Adapt the main path in ctrl/ParFlowCLMSetup.ksh (l. 17) and in ctrl/ParFlowCLM\_ini.txt (l.1)

To compile ParFlow, login on JUWELS Booster, go to the ctrl folder and simply do "./ParFlowCLMSetup.ksh" on a front node (has to be JUWELS Booster !)

For the preprocessing:
 login on JUWELS, go to ctrl: ./ParFlowCLMStarterJuwelsChainJob\_PrePro\_2.ksh -\> indicate the start date and the number of months in the script

For the simulations:
 login on JUWELS Booster, go to ctrl: ./ParFlowCLMStarterJuwelsChainJob\_RunSim\_2.ksh -\> indicate the start date and the number of months in the script. You can also indicate the job ID of a queuing job to extend the job chain without interruption

# 7)Validation

Use DE05\_validation and DE05\_validation\_hres\_c for validation of ParFlow run with HRES and HRES\_C respectively.

## 7.1)Validation of SM

Use SM\_ESACCI for each validation folder.For validation of HRES\_C, the following changes need to be made from the duplicate of HRES validation:

* Remove download.bash
* Remove dap.ceda.ac.uk (is not needed, can obtain from the other folder
* Remove ESACCI\_combined\*
* Remove ParFlowCLM\_volSM\_\*

Make the following changes in Extract\_regrid\_ParFlow\_VolSM\_clim.py:

* Change runname
* Copy ParFlowCLM\_DE05.out.0000.nc to the directory
* Change plotmonthly script (run name, datadir and file pattern).