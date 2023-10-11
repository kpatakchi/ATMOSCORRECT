# python packages and directories
from directories import *
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import Func
from mpl_toolkits.basemap import Basemap 
import numpy.ma as ma
import cartopy
import ftplib
from IPython.display import Image
from cartopy import crs as ccrs
from cartopy import feature
import cv2
import glob
import gc
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import shutil
import datashader as ds
import holoviews as hv
import datashader.transfer_functions as tf
from holoviews.operation.datashader import rasterize
import panel as pn
from bokeh.io import export_png
import netCDF4 as nc4
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import itertools

#------------------------#
#shutil.rmtree(DUMP_PLOT)
#os.mkdir(DUMP_PLOT)