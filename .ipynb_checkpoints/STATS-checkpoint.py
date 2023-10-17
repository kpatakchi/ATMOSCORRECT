from py_env_hpc import *
!pip install xskillscore
import xskillscore as xs

# define the parameters
parser = argparse.ArgumentParser(description="Settings for calculating stats")
parser.add_argument("--HPT_path", type=str, required=True, help="Which HPT path for results?")
parser.add_argument("--train_data_name", type=str, required=True, help="Which training data?")
parser.add_argument("--produce_data_name", type=str, required=True, help="Which produce data?")
parser.add_argument("--training_unique_name", type=str, required=True, help="Which training name?")
parser.add_argument("--produce_unique_name", type=str, required=True, help="Which produce name?")
parser.add_argument("--delete", type=str, required=True, help="Delete the existing STATS folder?")

args = parser.parse_args()

HPT_path=args.HPT_path
train_data_name = args.train_data_name
produce_data_name = args.produce_data_name
training_unique_name = args.training_unique_name
produce_unique_name = args.produce_unique_name
delete = args.delete

########## WARNING! Will Delete the STATS folder if delete==yes ##################
if delete=="True":
    !rm $STATS/*
########## WARNING! Will Delete the STATS folder if delete==yes ##################

# Define
HPT_PATH = PSCRATCH_DIR + HPT_path # directory to look for HPT results
predict_data_name = "predicted_for_" + produce_data_name[12:-4] + "_" + training_unique_name + ".npz"
date_start="2020-07-01T14"
date_end = "2023-04-26T23"
model_data="HRES"
reference_data="HSAF"
variable="pr"

### 0. Dataset Read/Write
#### 0.1 Read NetCDFs
# 1) Open the datasets:
print("0. Dataset Read/Write")

HRES = xr.open_dataset(f"{ATMOS_DATA}/HRES_pr.nc").sel(time=slice(date_start, date_end))
HSAF = xr.open_dataset(f"{ATMOS_DATA}/HSAF_pr.nc").sel(time=slice(date_start, date_end))
HRES_C = xr.open_dataset(f"{PREDICT_FILES}/HRES_C_{produce_data_name[12:-4]}_{training_unique_name}.nc").sel(time=slice(date_start, date_end))
# Replace precipitation values less than 0 with zero in HRES_C
HRES_C["pr"] = HRES_C["pr"].where(HRES_C["pr"] >= 0, 0)

#### 0.2 Write train_val_test arrays
#Prepares training, validation, and testing datasets from the best trained model.
# LOAD the data
train_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_train_indices.npy"
val_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_val_indices.npy"
produce_file = PRODUCE_FILES + "/" + produce_data_name
predict_file = PREDICT_FILES + "/" + predict_data_name
predicted_data = np.load(predict_file)

produce_file = np.load(produce_file)
mm_act = produce_file["canvas_y"][..., 0]
mask = produce_file["canvas_m"].astype(bool)[..., 0]
hres = produce_file["canvas_x"][..., 1]
days = produce_file["canvas_x"][..., 50, 50, 2]

mm_pred = predicted_data["Y_PRED"]
hres_c = hres - mm_pred
hres_c[hres_c < 0] = 0
hsaf = hres - mm_act

train_indices = np.load(train_ind_file)
val_indices = np.load(val_ind_file)
all_indices = np.arange(len(days))
test_indices = np.setdiff1d(all_indices, np.concatenate([train_indices, val_indices]))

def split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, indices, set_name):
    hsaf_out, mask_out, hres_out, days_out, hres_c_out, mm_act_out, mm_pred_out = (
        hsaf[indices], mask[indices], hres[indices], days[indices], hres_c[indices], 
        mm_act[indices], mm_pred[indices])

    hres_points = hres_out[mask_out]
    hres_c_points = hres_c_out[mask_out]
    mm_act_points = mm_act_out[mask_out]
    mm_pred_points = mm_pred_out[mask_out]
    hsaf_points = hsaf_out[mask_out]

    return {
        f"hsaf_{set_name}": hsaf_out,
        f"mask_{set_name}": mask_out,
        f"hres_{set_name}": hres_out,
        f"days_{set_name}": days_out,
        f"hres_c_{set_name}": hres_c_out,
        f"mm_act_{set_name}": mm_act_out,
        f"mm_pred_{set_name}": mm_pred_out,
        f"hres_{set_name}_points": hres_points,
        f"hres_c_{set_name}_points": hres_c_points,
        f"mm_act_{set_name}_points": mm_act_points,
        f"mm_pred_{set_name}_points": mm_pred_points,
        f"hsaf_{set_name}_points": hsaf_points}

# Splitting the data
train_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, train_indices, "train")
val_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, val_indices, "val")
test_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, test_indices, "test")

# Merging the data dictionaries
data = {**train_data, **val_data, **test_data}

# Save the arrays as a NumPy .npz file
np.savez(STATS + "/data_splits_" + HPT_path + ".npz", **data)

##### 0.2.1 Overall
# LOAD the data
train_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_train_indices.npy"
val_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_val_indices.npy"
produce_file = PRODUCE_FILES + "/" + produce_data_name
predict_file = PREDICT_FILES + "/" + predict_data_name
predicted_data = np.load(predict_file)

produce_file = np.load(produce_file)
mm_act = produce_file["canvas_y"][..., 0]
mask = produce_file["canvas_m"].astype(bool)[..., 0]
hres = produce_file["canvas_x"][..., 1]
days = produce_file["canvas_x"][..., 50, 50, 2]

mm_pred = predicted_data["Y_PRED"]
hres_c = hres - mm_pred
hres_c[hres_c < 0] = 0
hsaf = hres - mm_act

train_indices = np.load(train_ind_file)
val_indices = np.load(val_ind_file)
all_indices = np.arange(len(days))
test_indices = np.setdiff1d(all_indices, np.concatenate([train_indices, val_indices]))

def split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, indices, set_name):
    hsaf_out, mask_out, hres_out, days_out, hres_c_out, mm_act_out, mm_pred_out = (
        hsaf[indices], mask[indices], hres[indices], days[indices], hres_c[indices], 
        mm_act[indices], mm_pred[indices])

    hres_points = hres_out[mask_out]
    hres_c_points = hres_c_out[mask_out]
    mm_act_points = mm_act_out[mask_out]
    mm_pred_points = mm_pred_out[mask_out]
    hsaf_points = hsaf_out[mask_out]

    return {
        f"hsaf_{set_name}": hsaf_out,
        f"mask_{set_name}": mask_out,
        f"hres_{set_name}": hres_out,
        f"days_{set_name}": days_out,
        f"hres_c_{set_name}": hres_c_out,
        f"mm_act_{set_name}": mm_act_out,
        f"mm_pred_{set_name}": mm_pred_out,
        f"hres_{set_name}_points": hres_points,
        f"hres_c_{set_name}_points": hres_c_points,
        f"mm_act_{set_name}_points": mm_act_points,
        f"mm_pred_{set_name}_points": mm_pred_points,
        f"hsaf_{set_name}_points": hsaf_points}

# Splitting the data
train_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, train_indices, "train")
val_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, val_indices, "val")
test_data = split_data(hsaf, mask, hres, days, hres_c, mm_act, mm_pred, test_indices, "test")

# Merging the data dictionaries
data = {**train_data, **val_data, **test_data}

# Save the arrays as a NumPy .npz file
np.savez(STATS + "/data_splits_" + HPT_path + ".npz", **data)

##### 0.2.2 Seasonal
def get_season_indices(days):
    spring_indices = np.where((days >= 60) & (days <= 151))[0]
    summer_indices = np.where((days >= 152) & (days <= 243))[0]
    autumn_indices = np.where((days >= 244) & (days <= 334))[0]
    winter_indices = np.where((days >= 335) | (days <= 59))[0]
    return spring_indices, summer_indices, autumn_indices, winter_indices

spring_indices_train, summer_indices_train, autumn_indices_train, winter_indices_train = get_season_indices(data["days_train"])
spring_indices_val, summer_indices_val, autumn_indices_val, winter_indices_val = get_season_indices(data["days_val"])
spring_indices_test, summer_indices_test, autumn_indices_test, winter_indices_test = get_season_indices(data["days_test"])

seasons = ['spring', 'autumn', 'summer', 'winter']
kinds = ['train', 'val', 'test']

season_data = {}
for season in seasons:
    for kind in kinds:
        indices = globals()[f'{season}_indices_{kind}']
        season_data[f'hsaf_{kind}_{season}'] = data[f'hsaf_{kind}'][indices]
        season_data[f'mask_{kind}_{season}'] = data[f'mask_{kind}'][indices]
        season_data[f'hres_{kind}_{season}'] = data[f'hres_{kind}'][indices]
        season_data[f'days_{kind}_{season}'] = data[f'days_{kind}'][indices]
        season_data[f'hres_c_{kind}_{season}'] = data[f'hres_c_{kind}'][indices]
        season_data[f'mm_act_{kind}_{season}'] = data[f'mm_act_{kind}'][indices]
        season_data[f'mm_pred_{kind}_{season}'] = data[f'mm_pred_{kind}'][indices]
        
        mask_indices = data[f'mask_{kind}'][indices]

        season_data[f'hsaf_{kind}_{season}_points'] = data[f'hsaf_{kind}'][indices][mask_indices]
        season_data[f'mask_{kind}_{season}_points'] = data[f'mask_{kind}'][indices][mask_indices]
        season_data[f'hres_{kind}_{season}_points'] = data[f'hres_{kind}'][indices][mask_indices]
        season_data[f'hres_c_{kind}_{season}_points'] = data[f'hres_c_{kind}'][indices][mask_indices]
        season_data[f'mm_act_{kind}_{season}_points'] = data[f'mm_act_{kind}'][indices][mask_indices]
        season_data[f'mm_pred_{kind}_{season}_points'] = data[f'mm_pred_{kind}'][indices][mask_indices]
        
np.savez(STATS + "/data_splits_seasonal_" + HPT_path + ".npz", **season_data)


### 1. Resampled Data and Masks
#### 1.1. Resampled HRES/HSAF and no-NA Masks
print ("1. Resample Data and Masks")

# Resample HRES and HSAF datasets to daily temporal resolution
print("Resampling HRES and HSAF to daily...")
HRES_resampled_daily = HRES.resample(time="1D").sum(skipna=True)
HSAF_resampled_daily = HSAF.resample(time="1D").sum(skipna=True)
print("Resampling HRES and HSAF to daily complete.")

# Resample HRES and HSAF datasets to monthly temporal resolution
print("Resampling HRES and HSAF to monthly...")
HRES_resampled_monthly = HRES.resample(time="1M").sum(skipna=True)
HSAF_resampled_monthly = HSAF.resample(time="1M").sum(skipna=True)
print("Resampling HRES and HSAF to monthly complete.")

# Resample HRES_C dataset to daily temporal resolution
print("Resampling HRES_C to daily...")
HRES_C_resampled_daily = HRES_C.resample(time="1D").sum(skipna=True)
print("Resampling HRES_C to daily complete.")

# Resample HRES_C dataset to monthly temporal resolution
print("Resampling HRES_C to monthly...")
HRES_C_resampled_monthly = HRES_C.resample(time="1M").sum(skipna=True)
print("Resampling HRES_C to monthly complete.")

# Save the resampled datasets as NetCDF files
print("Saving resampled datasets...")
HRES_resampled_daily.to_netcdf(f"{STATS}HRES_pr_daily_" + HPT_path + ".nc")
HSAF_resampled_daily.to_netcdf(f"{STATS}HSAF_pr_daily_" + HPT_path + ".nc")
HRES_resampled_monthly.to_netcdf(f"{STATS}HRES_pr_monthly_" + HPT_path + ".nc")
HSAF_resampled_monthly.to_netcdf(f"{STATS}HSAF_pr_monthly_" + HPT_path + ".nc")
HRES_C_resampled_daily.to_netcdf(f"{STATS}HRES_C_pr_daily_" + HPT_path + ".nc")
HRES_C_resampled_monthly.to_netcdf(f"{STATS}HRES_C_pr_monthly_" + HPT_path + ".nc")
print("Resampled datasets saved.")

# Calculate masks
print("Calculating masks...")
mask_hourly = HSAF > -100
print("Calculating masks completed")

# Save the masks as NetCDF files
print("Saving masks...")
mask_hourly.to_netcdf(f"{STATS}nona_mask_pr_hourly_" + HPT_path + ".nc")
print("Masks saved.")


### 2. Error (mismatch):
print ("2. Calculate the Error")
# calculate:
import dask.array as da
HRES_da = da.from_array(HRES["pr"].data, chunks="auto")
HSAF_da = da.from_array(HSAF["pr"].data, chunks="auto")
error = HRES_da - HSAF_da
error_dataset = xr.Dataset({"error": (["time", "latitude", "longitude"], error.compute())})
output_path = STATS + "/error_" + HPT_path + ".nc"
error_dataset.to_netcdf(output_path)


### 3. ME, RMSE, and COR for 4 seasons and 3 temporal resolutions:
#### 3.1 For HRES
# Define the error metrics, temporal resolutions, and seasons
print("3. ME, RMSE, and COR for 4 seasons and 3 temporal resolutions:")
error_metrics = ["ME", "RMSE", "COR"]
temporal_resolutions = ["monthly", "daily", "hourly"]
seasons = ["DJF", "MAM", "JJA", "SON"]

# Load the resampled data
HRES_daily = xr.open_dataset(f"{STATS}HRES_pr_daily_" + HPT_path + ".nc")
HSAF_daily = xr.open_dataset(f"{STATS}HSAF_pr_daily_" + HPT_path + ".nc")
HRES_monthly = xr.open_dataset(f"{STATS}HRES_pr_monthly_" + HPT_path + ".nc")
HSAF_monthly = xr.open_dataset(f"{STATS}HSAF_pr_monthly_" + HPT_path + ".nc")

# Iterate through the combinations of error metrics, temporal resolutions, and seasons
for temporal_resolution in temporal_resolutions:
        if temporal_resolution == "daily":
            HRES_resampled = HRES_daily
            HSAF_resampled = HSAF_daily
            
        elif temporal_resolution == "monthly":
            HRES_resampled = HRES_monthly
            HSAF_resampled = HSAF_monthly
        else:
            HRES_resampled = HRES
            HSAF_resampled = HSAF
            
        for season in seasons:
            # Select the data for the specific season
            HRES_season = HRES_resampled.sel(time=HRES_resampled["time"].dt.season == season)
            HSAF_season = HSAF_resampled.sel(time=HSAF_resampled["time"].dt.season == season)

            for error_metric in error_metrics:
                print(f"Processing: {error_metric}_{temporal_resolution}_{season}")

                # Calculate the error metric
                if error_metric == "COR":
                    error = xs.pearson_r(HRES_season[variable], HSAF_season[variable], dim="time", skipna=True)
                elif error_metric == "ME":
                    error = HRES_season[variable] - HSAF_season[variable]
                    error = error.mean(dim="time", skipna=True)
                elif error_metric == "RMSE":
                    error = np.sqrt(((HRES_season[variable] - HSAF_season[variable]) ** 2).mean(dim="time", skipna=True))

                # Save the error as a NetCDF file
                file_name = f"{STATS}{season}_{temporal_resolution}_{error_metric}_" + HPT_path + ".nc"
                error.to_netcdf(file_name)

                print(f"Saved file: {file_name}")
                
#### 3.2 For HRES_C

# Define the error metrics, temporal resolutions, and seasons
error_metrics = ["ME", "RMSE", "COR"]
temporal_resolutions = ["monthly", "daily", "hourly"]
seasons = ["DJF", "MAM", "JJA", "SON"]

# Load the resampled data
HRES_C_daily = xr.open_dataset(f"{STATS}HRES_C_pr_daily_" + HPT_path + ".nc")
HSAF_daily = xr.open_dataset(f"{STATS}HSAF_pr_daily_" + HPT_path + ".nc")
HRES_C_monthly = xr.open_dataset(f"{STATS}HRES_C_pr_monthly_" + HPT_path + ".nc")
HSAF_monthly = xr.open_dataset(f"{STATS}HSAF_pr_monthly_" + HPT_path + ".nc")

# Iterate through the combinations of error metrics, temporal resolutions, and seasons
for temporal_resolution in temporal_resolutions:
    if temporal_resolution == "daily":
        HRES_C_resampled = HRES_C_daily
        HSAF_resampled = HSAF_daily
    elif temporal_resolution == "monthly":
        HRES_C_resampled = HRES_C_monthly
        HSAF_resampled = HSAF_monthly
    else:
        HRES_C_resampled = HRES_C
        HSAF_resampled = HSAF

    for season in seasons:
        # Select the data for the specific season
        HRES_C_season = HRES_C_resampled.sel(time=HRES_C_resampled["time"].dt.season == season)
        HSAF_season = HSAF_resampled.sel(time=HSAF_resampled["time"].dt.season == season)

        for error_metric in error_metrics:
            print(f"Processing: {error_metric}_{temporal_resolution}_{season}")

            # Calculate the error metric
            if error_metric == "COR":
                error = xs.pearson_r(HRES_C_season[variable], HSAF_season[variable], dim="time", skipna=True)
            elif error_metric == "ME":
                error = HRES_C_season[variable] - HSAF_season[variable]
                error = error.mean(dim="time", skipna=True)
            elif error_metric == "RMSE":
                error = np.sqrt(((HRES_C_season[variable] - HSAF_season[variable]) ** 2).mean(dim="time", skipna=True))

            # Save the error as a NetCDF file
            file_name = f"{STATS}{season}_{temporal_resolution}_{error_metric}_C_" + HPT_path + ".nc"
            error.to_netcdf(file_name)

            print(f"Saved file: {file_name}")

print("4. HRES-HSAF Correlations with time shifts:")
cor_stats = False
if cor_stats == True:
    ### 4. HRES-HSAF Correlations with time shifts
    #### 4.1. Daily
    from py_env_hpc import *
    import numpy as np
    import xarray as xr
    import xskillscore as xs

    model_data = "HRES"
    reference_data = "HSAF"
    date_start = "2020-07-01T13"
    date_end = "2021-07-01T12"
    variable = "pr"

    # Open the datasets
    HRES = xr.open_dataset(f"{ATMOS_DATA}/{model_data}_{variable}.nc").sel(time=slice(date_start, date_end))
    HSAF = xr.open_dataset(f"{ATMOS_DATA}/{reference_data}_{variable}.nc").sel(time=slice(date_start, date_end))

    # Define the time shifts
    time_shifts = range(-15, 14+1)  # -15 to +14

    # Calculate and save the correlations as npy matrices
    for shift in time_shifts:
        # Shift HRES
        HRES_shifted = HRES.shift(time=shift)
        HRES_shifted_daily = HRES_shifted.resample(time="D").sum()

        # Resample HSAF into daily data
        HSAF_daily = HSAF.resample(time="D").sum()

        # Calculate the correlation using xskillscore
        corr = xs.pearson_r(HRES_shifted_daily.pr, HSAF_daily.pr, dim="time", skipna=True)

        # Save the correlation matrix as npy
        np.save(f"{STATS}/DAILY_HSAF_HRES_COR_{date_start}_{date_end}_shift={shift}_" + HPT_path + ".npy", corr)
        
        #### 4.2. Monthly
        from py_env_hpc import *
        import numpy as np
        import xarray as xr
        import xskillscore as xs

        model_data = "HRES"
        reference_data = "HSAF"
        date_start = "2020-07-01T13"
        date_end = "2021-07-01T12"
        variable = "pr"

        # Open the datasets
        HRES = xr.open_dataset(f"{ATMOS_DATA}/{model_data}_{variable}.nc").sel(time=slice(date_start, date_end))
        HSAF = xr.open_dataset(f"{ATMOS_DATA}/{reference_data}_{variable}.nc").sel(time=slice(date_start, date_end))

        # Define the time shifts
        time_shifts = range(-15, 14+1)  # -15 to +14

        # Calculate and save the correlations as npy matrices
        for shift in time_shifts:
            # Shift HRES
            HRES_shifted = HRES.shift(time=shift)
            HRES_shifted_monthly = HRES_shifted.resample(time="M").sum()

            # Resample HSAF into daily data
            HSAF_daily = HSAF.resample(time="M").sum()

            # Calculate the correlation using xskillscore
            corr = xs.pearson_r(HRES_shifted_monthly.pr, HSAF_daily.pr, dim="time", skipna=True)

            # Save the correlation matrix as npy
            np.save(f"{STATS}/MONTHLY_HSAF_HRES_COR_{date_start}_{date_end}_shift={shift}_" + HPT_path + ".npy", corr)
            
print("5. HSAF Data Availability (DA) and Data Quality (DQ)")
DQ_DA_stats = False
if cor_stats == True:

    ### 5. HSAF Data Availability (DA) and Data Quality (DQ)
    #### 5.1. Seasonal DA and DQ

    import xarray as xr
    import numpy as np

    # Load the masks as xarray datasets
    mask_hourly = xr.open_dataset(f"{STATS}nona_mask_pr_hourly_" + HPT_path + ".nc")

    # Load the NetCDF file
    file_path = ATMOS_DATA + "/HSAF_pr.nc"
    data = xr.open_dataset(file_path)

    # Extract the qind variable
    qind = data["qind"]

    # Define the seasons
    seasons = ["DJF", "MAM", "JJA", "SON"]

    # Iterate over the seasons
    for season in seasons:
        # Select the data for the specific season
        mask_season = mask_hourly.sel(time=mask_hourly["time"].dt.season == season)
        qind_season = qind.sel(time=qind["time"].dt.season == season)

        # Calculate data availability for the season
        percent_available_season = (mask_season.sum("time") / mask_season.time.size) * 100

        # Calculate the average data quality index for the season
        avg_qind_season = qind_season.mean(dim="time", skipna=True)

        # Save data availability and data quality to NetCDF files
        percent_available_season.to_netcdf(f"{STATS}{season}_data_availability_" + HPT_path + ".nc")
        avg_qind_season.to_netcdf(f"{STATS}{season}_data_quality_" + HPT_path + ".nc")
        
        #### 5.2. DA and DQ over 2020, 2021, 2022, 2023
        
        # Load the masks as xarray datasets
        mask_hourly = xr.open_dataset(f"{STATS}nona_mask_pr_hourly_" + HPT_path + ".nc")

        # Load the NetCDF file
        file_path = ATMOS_DATA + "/HSAF_pr.nc"
        data = xr.open_dataset(file_path)

        # Extract the qind variable
        qind = data["qind"]

        # Define the years
        years = [2020, 2021, 2022, 2023]

        # Iterate over the years
        for year in years:
            # Select the data for the specific year
            mask_year = mask_hourly.sel(time=mask_hourly["time"].dt.year == year)
            qind_year = qind.sel(time=qind["time"].dt.year == year)

            # Calculate data availability for the year
            percent_available_year = (mask_year.sum("time") / mask_year.time.size) * 100

            # Calculate the average data quality index for the year
            avg_qind_year = qind_year.mean(dim="time", skipna=True)

            # Save data availability and data quality to NetCDF files
            percent_available_year.to_netcdf(f"{STATS}{year}_data_availability_" + HPT_path + ".nc")
            avg_qind_year.to_netcdf(f"{STATS}{year}_data_quality_" + HPT_path + ".nc")