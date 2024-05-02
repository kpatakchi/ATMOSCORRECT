from py_env_hpc import *

def datelist_generator (directory, dateformat, date_digit_start, date_digit_end):
    
    """ 
    returns a list of date_time objects from a list of files 
    in a directory.
    directory: the directory to the files
    dateformat: the format of the date in files; e.g., 
    "%Y%m%d_%H" for 20200101_00
    date_digit_start: the digid where the date starts;
    e.g., for h61_20201114_1500_01_fdk.nc the starting digit is 4
    date_digit_end: the digid where the date ends; e.g.,
    for h61_20201114_1500_01_fdk.nc the starting digit is 15
    """
    # sort the files
    files=sorted(os.listdir(directory))

    # create a unique list of all dates based on the given date format of the files
    file_list=list()
    for file in files:
        file=file[date_digit_start:date_digit_end]
        file_list.append(file)
    file_list=np.unique(file_list)
    file_list=file_list.tolist()

    # convert the format of the dates in the files into datetime format
    datelist = list()
    for fff in file_list:
        fdt = datetime.datetime.strptime(fff, dateformat)
        datelist.append(fdt)
    return datelist

def availability_plot (datelist, figsizex, figsizey, xlabinterval):
    
    """
    returns a barcode plot of data availability given the datelist.
    datelist is a list of datetime objects obtained by reading the
    data directory (see date_list_generator function).
    datelist: list of datetime objects
    figsizex, figsizey: size of the figure
    xlabinterval: the frequency of xlabel in terms of days.
    """
    
    A = np.ones((len(datelist)))
    pl.figure(figsize=(figsizex, figsizey), dpi=200)
    ax = pl.gca()
    pl.bar(datelist, A, color="black")
    pl.xticks(rotation=90)
    pl.yticks([])
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=xlabinterval))
    return pl.show()

def geo_idx(dd, dd_array):
    
    """
    search for nearest decimal degree in an array
    of decimal degrees and return the index.
    np.argmin returns the indices of minium
    value along an axis.
    so subtract dd from all values in dd_array,
    take absolute value and find index of 
    minium.
    can also deal with nan values!
    """
    geo_idx = np.nanargmin(np.abs(dd_array - dd))
    return geo_idx


def find_coord_ind(in_lat, in_lon, ll_or, acc, latitudevar, longitudevar):
    
    """
    search for a datapoint using lat/lon information given
    a certain accuracy. 
    
    """
    condition_lat=(ll_or[latitudevar]<in_lat+acc)*(ll_or[latitudevar]>in_lat-acc)
    condition_lon=(ll_or[longitudevar]<in_lon+acc)*(ll_or[longitudevar]>in_lon-acc)
    condition = condition_lat*condition_lon
    
    if np.nansum(condition)>2:
        print("accuracy is too low, try smaller numbers")
    if np.nansum(condition)==0: 
        print("accuracy is too high, try larger numbers")
    else: 
        latind=np.where(condition.values)[0][0]
        lonind=np.where(condition.values)[1][0]
        return latind, lonind

def nc_comparison_mapper (directory, files, titles, variable, projection, projection_name, dpi, fgx, fgy, nrows, ncols, tsteps, colormap, segments, vmax, savedirectory):
    from cartopy import feature

    """    
    plots comparison figures for 3d NetCDF files located in a directory
    directory: directory where the .nc files are located
    
    files: name of the files in the directory
    titles: the associated titles given in the plots for each .nc file
    variable: choose the variable to plot; e.g., pr for precipitation
    projection: type of projection
    projection_name: name of the projection as string
    dpi: the pixel density of the figure
    fgx: horizontal size of the figure
    fgy: vertical size of the figuer
    nrows: the number of rows in the comparison plot
    ncols: the number of columns in the comparison plot
    tsteps: the number of figures to generate based on the time.
    colormap: choose the colormap
    segmetns: assign how many segments the colormap is divided by
    vmax: the maximum value for the map
    savedirectory: where to save the image as .png file.
    """
    for time in range(tsteps):
        fig=pl.figure(figsize=(int(len(files)*ncols*fgx), int(len(files)*nrows*fgy)), dpi=dpi)
        for file_n in range(len(files)):
            ncfile=directory+"/"+files[file_n]
            title=titles[file_n]
            data=xarray.open_dataset(ncfile)[variable][time]
            timestring=np.datetime_as_string(data.time.values)[:16]
            cmap=pl.get_cmap(colormap, segments)
            if projection=="no_projection":
                ax=pl.subplot(nrows, ncols, file_n+1)
                im=ax.pcolormesh(data, cmap=cmap, vmax=vmax)
                if title=="raw":
                    pl.gca().invert_xaxis()
                else:
                    pl.gca().invert_yaxis()
                pl.colorbar(im, fraction=0.046, ticks=np.arange(0, vmax, vmax/segments))
                ax.set_title(title, pad=fgy*10) 
            else:
                ax=pl.subplot(nrows, ncols, file_n+1, projection=projection)
                if title=="raw":
                    lonvar="lon"
                    latvar="lat"
                else:
                    lonvar="longitude"
                    latvar="latitude"
                im=ax.pcolormesh(data[lonvar], data[latvar], data,
                                 transform=projection, cmap=cmap, vmax=vmax)
                ax.add_feature(feature.BORDERS)
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                ax.set_title(title, pad=fgy*10) 
                pl.colorbar(im, fraction=0.046, ticks=np.arange(0, vmax, vmax/segments))
                data.close()
        fig.suptitle(timestring + "_" + variable + "_" + projection_name)        
        pl.tight_layout()
        #pl.show()
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)
        pl.savefig(savedirectory+"/"+variable+"_"+projection_name+"_"+timestring+".png")
        pl.close("all")


def full_disc_mapper (directory, filenames, title, variable, dpi, fgx, fgy, colormap, segments, vmax, savedirectory):
    
    cmap=pl.get_cmap(colormap, segments)
    
    for file_n in range(len(files)):
        fig=pl.figure(figsize=(int(fgx), int(fgy)), dpi=dpi)
        ncfile=directory+"/"+files[file_n]
        title=title
        data=xarray.open_dataset(ncfile)[variable]
        im=pl.pcolormesh(data, cmap=cmap, vmax=vmax)
        pl.colorbar(im, fraction=0.046)
        fig.suptitle(files[file_n])        
        pl.tight_layout()
        #pl.show()
        pl.savefig(savedirectory+"/fig_"+files[file_n]+".png")
        pl.close("all")
        data.close()
        gc.collect()

        
def nc_mapper_2d (directory, filenames, time_str_start, time_str_stop, variable, lonvar, latvar, dpi, fgx, fgy, projection, grid_only, prj_name, colormap, segments, vmax, savedirectory):
    cmap=pl.get_cmap(colormap, segments)
    for file_n in range(len(filenames)):
        fig=pl.figure(figsize=(int(fgx), int(fgy)), dpi=dpi)
        ncfile=directory+"/"+filenames[file_n]
        data=xarray.open_dataset(ncfile)[variable]
        timestring=filenames[file_n][time_str_start:time_str_stop]
        title=filenames[file_n]+"_"+timestring+"_"+prj_name
        if projection=="no_projection":
            im=pl.pcolormesh(data, cmap=cmap, vmax=vmax)
            pl.colorbar(im, fraction=0.046)
        else:
            ax=pl.subplot(projection=projection)
            if grid_only==True:
                im=ax.pcolormesh(data[lonvar], data[latvar], data*np.nan,
                                 transform=projection, cmap=cmap, vmax=vmax)
            else:
                im=ax.pcolormesh(data[lonvar], data[latvar], data,
                                 transform=projection, cmap=cmap, vmax=vmax)
            ax.add_feature(feature.BORDERS)
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            pl.colorbar(im, fraction=0.046)
        fig.suptitle(title)
        pl.tight_layout()    
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)
        pl.savefig(savedirectory+"/fig_"+title+"_"+prj_name+".png")
        pl.close("all")
        data.close()
        gc.collect()
                 
        
def nc_mapper_3d (directory, filenames, variable, lonvar, latvar, timestep, dpi, fgx, fgy, projection, gridonly, prj_name, colormap, segments, vmax, savedirectory):
    cmap=pl.get_cmap(colormap, segments)
    for file_n in range(len(filenames)):
        for time in timestep:
            fig=pl.figure(figsize=(int(fgx), int(fgy)), dpi=dpi)
            ncfile=directory+"/"+filenames[file_n]
            data=xarray.open_dataset(ncfile)[variable][time]
            if gridonly==True:
                data=data.fillna(0)*0
            timestring=np.datetime_as_string(data["time"])[:16]
            title=filenames[file_n]+"_"+timestring+"_"+prj_name
            if projection=="no_projection":
                im=pl.pcolormesh(data, cmap=cmap, vmax=vmax)
                pl.colorbar(im, fraction=0.046)
            else:
                ax=pl.subplot(projection=projection)
                im=ax.pcolormesh(data[lonvar], data[latvar], data,
                                 transform=projection, cmap=cmap, vmax=vmax)
                ax.add_feature(feature.BORDERS)
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                pl.colorbar(im, fraction=0.046)
            fig.suptitle(title)
            pl.tight_layout()    
            if not os.path.exists(savedirectory):
                os.makedirs(savedirectory)
            pl.savefig(savedirectory+"/fig_"+title+"_"+prj_name+".png")
            pl.close("all")
            data.close()
            gc.collect()

            
def seasonal_map_histogram (seasons, variables, units, fx, fy, dpi, ncfiles,
                            titles, projection, projection_name, 
                            colormaps, segments, vmaxes, vmins, 
                            fontsize, minifontsize, n_bins, savedirectory):
    for var_n in range(len(variables)):
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)
        var=variables[var_n]
        unit=units[var_n]
        for season in seasons:
            fig = pl.figure(figsize=(fx, fy), dpi=dpi, facecolor="white")
            fig.suptitle(season, fontsize=fontsize, y=0.96, fontweight='bold')
            gridspec.GridSpec(3,6)
            row_col = (3,6)
            # DELTA HISTOGRAM
            ax0 = pl.subplot2grid(row_col, (2,0), colspan=6, rowspan=1)
            ax0.hist((ncfiles[2][season][var].mean(dim="time")).values.flatten(), density=True,
                     bins=n_bins, histtype="bar", color='mediumblue', range=(vmins[2], vmaxes[2]))
            ax0.set_ylabel("Probability", fontsize=fontsize)
            ax0.set_xlabel('Mismatch ' + unit, fontsize=fontsize)
            ax0.tick_params(labelsize=fontsize*.8) 
            ax0.grid(alpha=0.25)
            ax0.set_ylim(top=1.5)
            #SEASONAL MAPS
            for nnn in range(len(ncfiles)):
                ncfile=ncfiles[nnn][season][var].mean(dim="time")
                title=titles[nnn]
                vmin=vmins[nnn]
                vmax=vmaxes[nnn]
                colormap=colormaps[nnn]
                ax = pl.subplot2grid(row_col, (0,int(2*nnn)), colspan=2, rowspan=2, projection=projection)
                lonvar="longitude"
                latvar="latitude"
                im=ax.pcolormesh(ncfile[lonvar], ncfile[latvar], ncfile,
                                 transform=projection, cmap=colormap, vmax=vmax, vmin=vmin)
                ax.add_feature(feature.BORDERS)
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': minifontsize}
                gl.ylabel_style = {'size': minifontsize}
                inc=(vmax-vmin)/segments*2
                ax.set_title(title, pad=fy*5, fontsize=fontsize) 
                cb=pl.colorbar(im, fraction=0.046, 
                            ticks=np.arange(vmin, vmax+inc, inc)
                           , location="bottom")
                cb.ax.tick_params(labelsize=minifontsize)
            fig.tight_layout(pad=1.02)
            fig.savefig(savedirectory+"figure_" + season + "_" + var + ".png")
            fig.clf("all")
        # MERGE FIGURES INTO 1!
        figures=[]
        for season in seasons:
            image = Image.open(savedirectory+"figure_" + season + "_" + var + ".png")
            figures.append(image)
        # What is the size
        fig1=figures[0]
        fig1_size = fig1.size
        # Open the new image (figure)
        new_im = Image.new('RGB', (2*fig1_size[0],2*fig1_size[1]), (250,250,250))
        new_im.paste(figures[0], (0,0))
        new_im.paste(figures[1], (fig1_size[0],0))
        new_im.paste(figures[2], (0,fig1_size[1]))
        new_im.paste(figures[3], (fig1_size[0],fig1_size[1]))
        # SAVE THE NEW FIGURE
        new_im.save(savedirectory+"figure_"+var+".png")
        for season in seasons:
            os.remove(savedirectory+"figure_" + str(season) + "_" + var + ".png")
        print(var + " is plotted!")
        
        
def UNET(n_lat, n_lon, n_channels, ifn):
    
    import tensorflow as tf

    n_lat = n_lat
    n_lon = n_lon
    n_channels = n_channels # t-1, t, t+1
    ifn = ifn #initial feature number (number of initial filters)
    leakyrelu = tf.keras.layers.LeakyReLU()
    
    # Inputs
    inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
    inputs_bn = tf.keras.layers.BatchNormalization()(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(inputs_bn)
    #c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.BatchNormalization()(p1)

    c2 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(p1)
    #c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.BatchNormalization()(p2)

    c3 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(p2)
    #c3 = tf.keras.layers.Dropout(0.1)(c3)
    c3 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.BatchNormalization()(p3)

    c4 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(p3)
    #c4 = tf.keras.layers.Dropout(0.1)(c4)
    c4 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.BatchNormalization()(p4)

    c5 = tf.keras.layers.Conv2D(ifn*16, (3, 3), activation= leakyrelu, padding='same')(p4)
    #c5 = tf.keras.layers.Dropout(0.1)(c5)
    c5 = tf.keras.layers.Conv2D(ifn*16, (3, 3), activation= leakyrelu, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(ifn*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(u6)
    #c6 = tf.keras.layers.Dropout(0.1)(c6)
    c6 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)

    u7 = tf.keras.layers.Conv2DTranspose(ifn*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(u7)
    #c7 = tf.keras.layers.Dropout(0.1)(c7)
    c7 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    u8 = tf.keras.layers.Conv2DTranspose(ifn*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(u8)
    #c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)

    u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(u9)
    #c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


def square_maker(X_TRAIN, Y_TRAIN, MASK, xleft, ydown, nx, ny):

    canvas_x=np.zeros((X_TRAIN.shape[0], nx, ny, X_TRAIN.shape[3]))
    canvas_y=np.zeros((Y_TRAIN.shape[0], nx, ny, Y_TRAIN.shape[3]))
    canvas_m=np.zeros((MASK.shape[0], nx, ny, MASK.shape[3]))
    
    canvas_x[:]=0
    canvas_y[:]=0
    canvas_m[:]=0

    canvas_x[:, xleft:125+xleft, ydown:196+ydown, :]=X_TRAIN
    canvas_y[:, xleft:125+xleft, ydown:196+ydown, :]=Y_TRAIN
    canvas_m[:, xleft:125+xleft, ydown:196+ydown, :]=MASK
    
    return canvas_x, canvas_y, canvas_m

def de_square(canvas_y, xleft, ydown, nx, ny):

    y=canvas_y[:, xleft:125+xleft, ydown:196+ydown, 0]
    
    return y