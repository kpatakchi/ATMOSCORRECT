from py_env_train import *

def UNET_ATT(n_lat, n_lon, n_channels, ifn):
    import tensorflow as tf

    n_lat = n_lat
    n_lon = n_lon
    n_channels = n_channels  # t-1, t, t+1
    ifn = ifn  # initial feature number (number of initial filters)
    leakyrelu = tf.keras.layers.LeakyReLU()
    dropout_rate=0.2

    # Attention block
    def attention_block(input_tensor, filters):
        se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        se = tf.keras.layers.Conv2D(filters // 8, (1, 1), activation='relu', padding='same')(se)
        se = tf.keras.layers.Conv2D(filters, (1, 1), activation='sigmoid', padding='same')(se)
        return tf.keras.layers.Multiply()([input_tensor, se])

    # Inputs
    inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
    inputs_bn = tf.keras.layers.BatchNormalization()(inputs)

    # Contraction path
    c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs_bn)
    c1 = attention_block(c1, ifn)  # Add attention block here
    c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
    c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.BatchNormalization()(p1)

    c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
    c2 = attention_block(c2, ifn * 2)  # Add attention block here
    c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
    c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.BatchNormalization()(p2)

    c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
    c3 = attention_block(c3, ifn * 4)  # Add attention block here
    c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
    c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.BatchNormalization()(p3)

    c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
    c4 = attention_block(c4, ifn * 8)  # Add attention block here
    c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
    c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.BatchNormalization()(p4)

    c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(p4)
    c5 = attention_block(c5, ifn * 16)  # Add attention block here
    c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
    c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(ifn * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(u6)
    c6 = attention_block(c6, ifn * 8)  # Add attention block here
    c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
    c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)

    u7 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u7)
    c7 = attention_block(c7, ifn * 4)  # Add attention block here
    c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
    c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    u8 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u8)
    c8 = attention_block(c8, ifn * 2)  # Add attention block here
    c8 = tf.keras.layers.Dropout(dropout_rate)(c8)  # Add dropout layer here
    c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)

    u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u9)
    c9 = attention_block(c9, ifn)  # Add attention block here
    c9 = tf.keras.layers.Dropout(dropout_rate)(c9)  # Add dropout layer here
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model

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
    #c1 = tf.keras.layers.Dropout(0.5)(c1)
    c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.BatchNormalization()(p1)

    c2 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(p1)
    #c2 = tf.keras.layers.Dropout(0.5)(c2)
    c2 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.BatchNormalization()(p2)

    c3 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(p2)
    #c3 = tf.keras.layers.Dropout(0.5)(c3)
    c3 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.BatchNormalization()(p3)

    c4 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(p3)
    #c4 = tf.keras.layers.Dropout(0.5)(c4)
    c4 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.BatchNormalization()(p4)

    c5 = tf.keras.layers.Conv2D(ifn*16, (3, 3), activation= leakyrelu, padding='same')(p4)
    #c5 = tf.keras.layers.Dropout(0.5)(c5)
    c5 = tf.keras.layers.Conv2D(ifn*16, (3, 3), activation= leakyrelu, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(ifn*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(u6)
    #c6 = tf.keras.layers.Dropout(0.5)(c6)
    c6 = tf.keras.layers.Conv2D(ifn*8, (3, 3), activation= leakyrelu, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)

    u7 = tf.keras.layers.Conv2DTranspose(ifn*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(u7)
    #c7 = tf.keras.layers.Dropout(0.5)(c7)
    c7 = tf.keras.layers.Conv2D(ifn*4, (3, 3), activation= leakyrelu, padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)

    u8 = tf.keras.layers.Conv2DTranspose(ifn*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(u8)
    #c8 = tf.keras.layers.Dropout(0.5)(c8)
    c8 = tf.keras.layers.Conv2D(ifn*2, (3, 3), activation= leakyrelu, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)

    u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(u9)
    #c9 = tf.keras.layers.Dropout(0.5)(c9)
    c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation= leakyrelu, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

def make_canvas(data, canvas_shape, trim=True):
    """
    Pads the input data with zeros to create a canvas of the specified shape
    while keeping the original data centered or trimmed to fit the canvas. 

    Args:
    - data (numpy.ndarray): Input data with shape (num_samples, original_dim1, original_dim2, channels)
    - canvas_shape (tuple): Desired shape of the canvas in the format (canvas_dim1, canvas_dim2)
    - trim (bool): If True, trims the original data to fit the canvas instead of making a bigger canvas.

    Returns:
    - numpy.ndarray: Canvas with shape (num_samples, canvas_dim1, canvas_dim2, channels)
    """
    num_samples, original_dim1, original_dim2, channels = data.shape
    canvas_dim1, canvas_dim2 = canvas_shape

    # If trim is True and the original data is larger than the canvas, trim the data
    if trim and (original_dim1 > canvas_dim1 or original_dim2 > canvas_dim2):
        start_idx1 = (original_dim1 - canvas_dim1) // 2
        start_idx2 = (original_dim2 - canvas_dim2) // 2
        data = data[:, start_idx1:start_idx1+canvas_dim1, start_idx2:start_idx2+canvas_dim2, :]

    # If trim is False or the original data is smaller than the canvas, pad the data
    else:
        # Calculate the difference between the original dimensions and the canvas dimensions
        diff_dim1 = canvas_dim1 - original_dim1
        diff_dim2 = canvas_dim2 - original_dim2

        # Calculate the padding for the top and left sides
        top_pad = diff_dim1 // 2
        left_pad = diff_dim2 // 2

        # Calculate the padding for the bottom and right sides
        bottom_pad = diff_dim1 - top_pad
        right_pad = diff_dim2 - left_pad

        # Create a new array of zeros with the shape of the canvas
        canvas = np.zeros((num_samples, canvas_dim1, canvas_dim2, channels))

        # Insert the original data in the center of the canvas
        canvas[:, top_pad:top_pad+original_dim1, left_pad:left_pad+original_dim2, :] = data
        return canvas

def create_calendar (date_start, date_end, MODEL_shape, daily=True):
    
    """
    Creates a numpy array in the same shape as MODEL (model-based data) representing a calendar.
    CAL[..., 0] contains the day of the year for each day inside a year.
    CAL[..., 1] contains the year number.
    
    Args:
    - date_start (str): Start date of the calendar in YYYY-MM-DD format.
    - date_end (str): End date of the calendar in YYYY-MM-DD format.
    - daily (bool): Whether the data is daily or hourly. If hourly, each day will contain 24 hours.
    
    Returns:
    - CAL (np.ndarray): Numpy array of shape (n_days, LAT, LON, 2) representing the calendar.
    """

    CAL = np.ones((MODEL_shape[0], MODEL_shape[1], MODEL_shape[2], 2), dtype="int32")    
    y_start = int(date_start[:4])
    y_end = int(date_end[:4])
    days_in_year = 365
    hours_in_day = 24

    if daily:
        for year in range(y_start, y_end+1):
            for day in range(days_in_year):
                dayloc=(year-y_start)*days_in_year+day-1
                CAL[dayloc, ..., 0] = year
                CAL[dayloc, ..., 1] = day + 1
    else: 
        for year in range(y_start, y_end+1):
            for day in range(days_in_year):
                for hour in range (hours_in_day):
                    hourlock=(year-y_start)*days_in_year*hours_in_day+day*hours_in_day-1
                    CAL[hourlock, ..., 0] = year
                    CAL[hourlock, ..., 1] = day + 1
                    
    return CAL

def spatiodataloader (directory, MODEL_shape):
    
    """
    Loads spatial data from a NumPy file and reshapes it to the specified dimensions.

    Parameters:
        directory (str): Path and filename of the NumPy binary file containing the spatial data.
        MODEL_shape (tuple): Desired dimensions of the output array. A 3-tuple specifying the height, width, and number of channels (e.g., color channels) of the output array.

    Returns:
        A NumPy array of shape MODEL_shape that contains the loaded spatial data.

    """
    SP = np.load(directory)["arr_0"]
    SP_reshaped=np.empty((MODEL_shape[0], MODEL_shape[1], MODEL_shape[2], SP.shape[2]))
    
    for x in range (MODEL_shape[1]):
        for y in range (MODEL_shape[2]):
            SP_reshaped[:, x, y, 0] = SP[x, y, 0]
            SP_reshaped[:, x, y, 1] = SP[x, y, 1]
            SP_reshaped[:, x, y, 2] = SP[x, y, 2]

    return SP_reshaped


def calculate_channels(n_ensembles, task_name, laginensemble=False):
    """
    Calculate the number of channels based on the given task name and number of ensembles.

    Args:
    - n_ensembles: int, number of ensembles.
    - task_name: str, name of the task.
    - laginensemble: bool, optional, whether the ensembles are lagged or not. Default is False.

    Returns:
    - n_channels: int, number of channels.

    Raises:
    - ValueError: if an invalid task name is given.

    Task Name	Explanation	Name	N_channels
    A1	Model-only for t=t	“model-only”	N_ensembles
    A2	Model-only for t=t and t=t-1	“model-lag”	N_ensembles*2
    A3	Same as A2, including the day of the year and the year information. 	“temporal”	N_ensembles+2 (if laginensemble=False), N_ensembles*2 (if laginensemble=True)
    A4	Same as A3, but also including the lat/lon/altitude information.	“spatiotemporal”	N_ensembles+5 (if laginensemble=False), N_ensembles*2+5 (if laginensemble=True)
    A5	Same as A2, but also including the lat/lon/altitude iformation	“spatial”	N_ensembles+3 (if laginensemble=False), N_ensembles*2+3 (if laginensemble=True)
    """
    if task_name == "model-only":
        n_channels = n_ensembles
    elif task_name == "model-lag":
        n_channels = n_ensembles * 2
    elif task_name == "temporal":
        n_channels = n_ensembles + 2 if not laginensemble else n_ensembles * 2
    elif task_name == "spatiotemporal":
        n_channels = n_ensembles + 5 if not laginensemble else n_ensembles * 2 + 5
    elif task_name == "spatial":
        n_channels = n_ensembles + 3 if not laginensemble else n_ensembles * 2 + 3
    else:
        raise ValueError("Invalid task name: " + task_name)
        
    return n_channels

def data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble):
    # The following is defined automatically:
    n_ensembles = len(model_data)
    n_channels = calculate_channels(n_ensembles, task_name, laginensemble=laginensemble)
    if reference_data == ["COSMO_REA6"]:
        canvas_size = (400, 400) 
        topo_dir = '/p/project/deepacf/kiste/patakchiyousefi1/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
        trim = True
        daily = True
    elif reference_data == ["HSAF"]:
        topo_dir = '/p/project/deepacf/kiste/patakchiyousefi1/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
        canvas_size = (128, 256)
        trim = False
        daily = False
    else:
        topo_dir = ''  # Set the default value if reference_data is not recognized
        canvas_size = (0, 0)
        trim = False
        daily = False
    
    data_unique_name = f"train_data{'_daily' if daily else '_hourly'}_{variable}_{model_data}_{reference_data}_{mm}_{n_channels}_{'laginensemble' if laginensemble else ''}_{task_name}_{'.'.join(map(str, canvas_size))}_{date_start}_{date_end}_{mask_type}"
    filename = f"{data_unique_name}.npz"
    return filename

def generate_training_unique_name(loss, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs):
    training_unique_name = loss + "_" + str(Filters) + "_" + str(LR) + "_" + str(min_LR) + "_" + str(lr_factor) + "_" + str(lr_patience) + "_" + str(BS) + "_" + str(patience) + "_" + str(val_split) + "_" + str(epochs)
    return training_unique_name

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
        print("Opening datasets...")
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
            canvas_m = np.ones_like(canvas_y)  
            canvas_m[canvas_y == -999] = 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land
            # to remove the zeros out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0
                
        if mask_type == "no_na_intensity":
            TRUTH = Y_TRAIN - X_TRAIN #reference(to be used in intensity weights)
            canvas_t = make_canvas(TRUTH, canvas_size, trim)
            greater_zero = canvas_t[..., 0]>=0
            less_pointone = canvas_t[..., 0]<0.1
            greater_pointone = canvas_t[..., 0]>=0.1
            less_twohalf = canvas_t[..., 0]<2.5
            greater_twohalf = canvas_t[..., 0]>=2.5
            dry=greater_zero*less_pointone
            light=greater_pointone*less_twohalf
            heavy=greater_twohalf

            canvas_m = np.ones_like(canvas_y) 
            canvas_m[canvas_y == -999] = 0 #replace nan with 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land #only on land
            
            # to remove the ones out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0.
                
            canvas_m[dry] *= 0.01  # Multiply by 0.01 in dry conditions
            canvas_m[light] *= 0.04  # Multiply by 0.04 in light conditions
            canvas_m[heavy] *= 0.95  # Multiply by 0.95 in heavy conditions
                

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
        

def generate_produce_unique_name(loss, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs):
    training_unique_name = loss + "_" + str(Filters) + "_" + str(LR) + "_" + str(min_LR) + "_" + str(lr_factor) + "_" + str(lr_patience) + "_" + str(BS) + "_" + str(patience) + "_" + str(val_split) + "_" + str(epochs)
    return training_unique_name

def prepare_produce(PPROJECT_DIR, PRODUCE_FILES, ATMOS_DATA, filename, model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble):
    
    """
    This function prepares the production data for UNET model.
    
    Args:
        PPROJECT_DIR (str): The project directory path.
        PRODUCE_FILES (str): The directory where the production files will be saved.
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
    file_name_pro = "produce_for_" + filename
    
    if file_name_pro not in os.listdir(PRODUCE_FILES):
        
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
                
                
        if mask_type == "no_na_intensity":
            TRUTH = Y_TRAIN - X_TRAIN #reference(to be used in intensity weights)
            canvas_t = make_canvas(TRUTH, canvas_size, trim)
            greater_zero = canvas_t[..., 0]>=0
            less_pointone = canvas_t[..., 0]<0.1
            greater_pointone = canvas_t[..., 0]>=0.1
            less_twohalf = canvas_t[..., 0]<2.5
            greater_twohalf = canvas_t[..., 0]>=2.5
            dry=greater_zero*less_pointone
            light=greater_pointone*less_twohalf
            heavy=greater_twohalf

            canvas_m = np.ones_like(canvas_y) 
            canvas_m[canvas_y == -999] = 0 #replace nan with 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land #only on land
            
            # to remove the ones out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0.
                
            canvas_m[dry] *= 0.01  # Multiply by 0.01 in dry conditions
            canvas_m[light] *= 0.04  # Multiply by 0.04 in light conditions
            canvas_m[heavy] *= 0.95  # Multiply by 0.95 in heavy conditions
            

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
        
        canvas_x = canvas_x.astype(np.float32)
        canvas_y = canvas_y.astype(np.float32)
        canvas_m = canvas_m.astype(np.float32)
        
        X_TRAIN_tminus = None
        CAL = None
        SPP = None
        
        np.savez(PRODUCE_FILES + "/produce_for_" + filename, canvas_x=canvas_x, canvas_y=canvas_y, canvas_m=canvas_m)

        print("Data generated")
    else:
        print("The data with the same unique name is already available")
        

def unmake_canvas(canvas, original_shape):
    """
    Restores the original data from a canvas by removing zero-padding and optionally trimming.

    Args:
    - canvas (numpy.ndarray): Input canvas data with shape (num_samples, canvas_dim1, canvas_dim2, channels)
    - original_shape (tuple): Original shape of the data in the format (original_dim1, original_dim2)
    Returns:
    - numpy.ndarray: Restored data with shape (num_samples, original_dim1, original_dim2)
    """
    num_samples, canvas_dim1, canvas_dim2 = canvas.shape
    original_dim1, original_dim2 = original_shape
    
    diff_dim1 = canvas_dim1 - original_dim1
    diff_dim2 = canvas_dim2 - original_dim2
    
    # Calculate the padding for the top and left sides
    top_pad = diff_dim1 // 2
    left_pad = diff_dim2 // 2

    # Calculate the padding for the bottom and right sides
    bottom_pad = diff_dim1 - top_pad
    right_pad = diff_dim2 - left_pad
    
    # Remove zero-padding from the canvas to restore the original data
    data = canvas[:, top_pad:top_pad+original_dim1, left_pad:left_pad+original_dim2]
    return data

def de_prepare_produce(Y_PRED, PREDICT_FILES, ATMOS_DATA, filename, model_data, date_start, date_end, variable, training_unique_name, reference_data, onedelay):
        
    import xarray as xr
    import pandas as pd
    
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
    
    if onedelay==True:
        date_start = (pd.to_datetime(date_start) + pd.DateOffset(hours=1)).strftime("%Y-%m-%dT%H")
    
    # Open the first model in model_data
    model = xr.open_dataset(f"{ATMOS_DATA}/{model_data[0]}_{variable}.nc")
    model = model[variable].sel(time=slice(date_start, date_end))

    # Retrieve lat and lon shape from the model
    lat_shape = model.latitude.shape[0]
    lon_shape = model.longitude.shape[0]
    
    # Load topographical data
    SPP = spatiodataloader(topo_dir, model.shape)
    SPP_canvas = make_canvas(SPP, canvas_size, trim)

    # Define ocean mask
    land= SPP_canvas[..., 2]>0
    land = land * 1
    
    # mismatch predictions for ocean is removed. 
    Y_PRED = Y_PRED * land

    # Restore the original shape of Y_PRED using unmake_canvas function
    Y_PRED = unmake_canvas(Y_PRED, (lat_shape, lon_shape))

    # Subtract Y_PRED from model
    diff = model - Y_PRED

    # Save the result in a NETCDF file
    data_unique_name = filename[:-4]
    output_filename = f"{PREDICT_FILES}/HRES_C_{data_unique_name}_{training_unique_name}.nc"
    diff.to_netcdf(output_filename)
