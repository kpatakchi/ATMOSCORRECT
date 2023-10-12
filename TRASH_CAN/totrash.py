def prepare_train(PPROJECT_DIR, TRAIN_FILES, ATMOS_DATA, filename, model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble, val_split):
    
    """
    This function prepares the training data for UNET model.
    
    Args:
        PPROJECT_DIR (str): The project directory path.
        TRAIN_FILES (str): The directory where the training files will be saved.
        ATMOS_DATA (str): The directory containing the atmospheric data.
        filename (str): The name of the file to be saved.
        model_data (list): A list of model names.
        reference_data (list): A list of reference data names.
        task_name (str): The type of task for the model.
        mm (str): The type of target (mismatch or direct).
        date_start (str): The start date for selecting the data.
        date_end (str): The end date for selecting the data.
        variable (str): The variable to be used in the data.
        mask_type (str): The type of mask to be applied.
        laginensemble (int): The lag in the ensemble dimension.
        val_split (float): The validation data split ratio.
    """
        
    import os
    import xarray as xr
    
    data_unique_name=filename[:-4]
    
    if filename not in os.listdir(TRAIN_FILES):
        
        # 1) Open the datasets:
        datasets = []
        for model in model_data:
            dataset = xr.open_dataset(f"{ATMOS_DATA}/{model}_{variable}.nc")
            dataset = dataset[variable].sel(time=slice(date_start, date_end))
            datasets.append(dataset)

        REFERENCE = xr.open_dataset(f"{ATMOS_DATA}/{reference_data[0]}_{variable}.nc")
        REFERENCE = REFERENCE[variable].sel(time=slice(date_start, date_end))

        # 2) Calculate the calendar data according to REFERENCE (starting the calendar one day later)
        dayofyear = REFERENCE[1:, ...].time.dt.dayofyear.values
        dayofyear_resh = np.tile(dayofyear[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        yeardate = REFERENCE[1:, ...].time.dt.year.values
        yeardate_resh = np.tile(yeardate[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        CAL = np.stack((dayofyear_resh, yeardate_resh), axis=3)

        REFERENCE = REFERENCE.values[:, :, :, np.newaxis]  # add new axis along ensemble dimension
        datasets = [dataset.values for dataset in datasets]
        MODEL = np.stack(datasets, axis=-1)

        # 2) Define the Target (mismatch or direct):
        print("Defining the target...")
        if len(datasets) > 1:
            TARGET = (MODEL[0] - REFERENCE) if (mm == "MM") else REFERENCE
        else:
            TARGET = (MODEL - REFERENCE) if (mm == "MM") else REFERENCE
        if MODEL.shape[0] < 1:
            print("The selected dates don't exist in the netcdf files!")

        if reference_data == ["COSMO_REA6"]:
            canvas_size = (400, 400) 
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
            trim=True
            daily=True
        if reference_data == ["HSAF"]:
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
            canvas_size = (128, 256)
            trim=False
            daily=False

        # Close the netCDF files and release memory
        datasets = None
        REFERENCE = None
        dayofyear = None
        yeardate = None
        yeardate_resh = None

        # 3) Define X_Train and Y_Train
        print("Defining X_Train and Y_Train...")
        Y_TRAIN = TARGET[1:, ...]  # t
        X_TRAIN = MODEL[1:, ...]  # t
        X_TRAIN_tminus = MODEL[:-1, ...]
        canvas_y = make_canvas(Y_TRAIN, canvas_size, trim)
        canvas_y = np.nan_to_num(canvas_y, nan=-999)  # fill values
        SPP = spatiodataloader(topo_dir, X_TRAIN.shape)

        if mask_type == "no_na":
            canvas_m = np.ones_like(canvas_y)  # mask for na values (-9999)
            canvas_m[canvas_y == -999] = 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land
            # to remove the zeros out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0

        if task_name == "model_only":
            X_TRAIN = X_TRAIN

        if task_name == "model-lag":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN), axis=3)

        if task_name == "temporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL), axis=3)

        if task_name == "spatial":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, SPP), axis=3)

        if task_name == "spatiotemporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL, SPP), axis=3)

        canvas_x = make_canvas(X_TRAIN, canvas_size, trim)
            
        X_TRAIN_tminus = None
        CAL = None
        SPP = None

        # Save train and validation data
        print("Saving train and validation data...")
        np.random.seed(hash(data_unique_name) % 2**32 - 1)
        num_samples = canvas_x.shape[0]
        indices = np.arange(num_samples)  # Create an array of indices

        train_prop = 1 - val_split
        num_train_samples = int(np.round(num_samples * train_prop))

        # Create clusters of 10 consecutive numbers for validation
        cluster_size = 10
        num_clusters = num_samples // cluster_size
        num_val_clusters = int(np.ceil(num_clusters * val_split))
        val_clusters = np.random.choice(num_clusters, size=num_val_clusters, replace=False)

        val_indices = []
        for cluster in val_clusters:
            start_index = cluster * cluster_size
            end_index = start_index + cluster_size
            val_indices.extend(list(range(start_index, end_index)))

        val_indices = np.sort(np.array(val_indices))
        train_indices = np.setdiff1d(indices, val_indices)

        train_x = canvas_x[train_indices].astype(np.float32)
        train_y = canvas_y[train_indices].astype(np.float32)
        train_m = canvas_m[train_indices].astype(np.float32)
        val_x = canvas_x[val_indices].astype(np.float32)
        val_y = canvas_y[val_indices].astype(np.float32)
        val_m = canvas_m[val_indices].astype(np.float32)

        canvas_y = None
        canvas_x = None
        canvas_m = None

        # Save as float32 files
        np.savez(TRAIN_FILES + "/" + filename,
                 train_x=train_x,
                 train_y=train_y,
                 train_m=train_m,
                 val_x=val_x,
                 val_y=val_y,
                 val_m=val_m)
        
        train_x = None
        train_y = None
        train_m = None
        val_x = None
        val_y = None
        val_m = None

        np.save(PPROJECT_DIR+'/AI MODELS/00-UNET/'+data_unique_name+"_train_indices.npy", train_indices)
        np.save(PPROJECT_DIR+'/AI MODELS/00-UNET/'+data_unique_name+"_val_indices.npy", val_indices)

        print("Data generated")
    else:  
        print("Data is available already")