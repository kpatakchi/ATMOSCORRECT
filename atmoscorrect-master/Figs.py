from py_env_hpc import *

# Define the setup:

HPT_PATH = PSCRATCH_DIR + "HPT_v1/" # directory to look for HPT results
train_data_name = "train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-03-26T23_no_na.npz" # name of the training data used for training
produce_data_name = "produce_for_train_data_hourly_pr_['HRES']_['HSAF']_MM_6__spatiotemporal_128.256_2020-07-01T13_2023-04-26T23_no_na.npz" # name of the production file name used for production

import pandas as pd
import glob

csv_files = glob.glob(HPT_PATH + "*.csv")

# Create lists to store the extracted hyperparameters, stop_epoch, and avg_loss
LR_list = []
BS_list = []
LRF_list = []
IFN_list = []
stop_epoch_list = []
avg_loss_list = []
val_loss_list = []
stop_loss_list = []

# Iterate over each CSV file
for file in csv_files:
    data = pd.read_csv(file)
    
    training_unique_name = file.replace(HPT_PATH, '').replace('.csv', '')
    hyperparameters = training_unique_name.split('_')
    
    LR = float(hyperparameters[2])
    BS = int(hyperparameters[6])
    LRF = float(hyperparameters[4])
    IFN = int(hyperparameters[1])
    
    val_loss = data['val_mse']
    min_val_loss = val_loss.min()
    min_val_loss_index = val_loss.idxmin()
    loss = data['mse']
    stop_loss = loss[min_val_loss_index]
    # Calculate the average of min_val_loss and stop_loss
    avg_loss = (min_val_loss * 0.1 + stop_loss * 0.9)
    
    # Extract the stop epoch as index of the minimum val_loss + 1
    stop_epoch = min_val_loss_index+1
    
    # Append the extracted hyperparameters, stop_epoch, and avg_loss to the lists
    LR_list.append(LR)
    BS_list.append(BS)
    IFN_list.append(IFN)
    stop_epoch_list.append(stop_epoch)
    avg_loss_list.append(avg_loss)
    val_loss_list.append(min_val_loss)
    stop_loss_list.append(stop_loss)

# Create a DataFrame with the extracted data
data_df = pd.DataFrame({
    'LR': LR_list,
    'BS': BS_list,
    'IFN': IFN_list,
    'val_loss': val_loss_list,
    'stop_loss': stop_loss_list,
    'avg_loss': avg_loss_list,
    'stop_epoch': stop_epoch_list  # Add the stop_epoch column as the index + 1
})

top_36_data_df = data_df.nsmallest(36, 'val_loss')
top_36_data_df = top_36_data_df.rename(columns={'stop_loss': 'train_loss'})
top_36_data_df = top_36_data_df.reset_index(drop=True)
file_path = DUMP_PLOT + "/top_36_data_df.xlsx"
top_36_data_df['rank'] = range(1, 37)
# Reorder the DataFrame columns
columns = ['rank', 'LR', 'BS', 'IFN', 'val_loss', 'train_loss', 'avg_loss', 'stop_epoch']
top_36_data_df = top_36_data_df[columns]

# Save DataFrame to an Excel file
top_36_data_df.to_excel(file_path, index=False)

best_index = val_loss_list.index(min(val_loss_list))
best_csv_file = csv_files[best_index]
best_csv_filename = os.path.basename(best_csv_file[:-4])
#print("BEST training_unique_name:", best_csv_filename)
training_unique_name = best_csv_filename
best_epoch=top_36_data_df["stop_epoch"][0]

top_36_data_df

import itertools

predict_data_name = "predicted_for_" + produce_data_name[12:-4] + "_" + training_unique_name + ".npz"

# orig_train_val_data_name
train_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_train_indices.npy"
val_ind_file = PPROJECT_DIR + '/AI MODELS/00-UNET/' + train_data_name[:-4] + "_val_indices.npy"
ori_file = PRODUCE_FILES + "/" + produce_data_name
predict_file = PREDICT_FILES + "/" + predict_data_name

ori_data = np.load(ori_file)
data_y = ori_data["canvas_y"]
data_m = ori_data["canvas_m"].astype(bool)

train_indices = np.load(train_ind_file)
val_indices = np.load(val_ind_file)
all_indices = np.arange(len(data_y))
test_indices = np.setdiff1d(all_indices, np.concatenate([train_indices, val_indices]))

data_m_train = data_m[train_indices]
data_m_val = data_m[val_indices]
data_m_test = data_m[test_indices]

train_y = data_y[train_indices][data_m_train]
val_y = data_y[val_indices][data_m_val]
test_y = data_y[test_indices][data_m_test]

predicted_data = np.load(predict_file)
pred_y = predicted_data["Y_PRED"]

train_pred_y = pred_y[train_indices][data_m_train[..., 0]]
val_pred_y = pred_y[val_indices][data_m_val[..., 0]]
test_pred_y = pred_y[test_indices][data_m_test[..., 0]]

# to be adjusted:
sample_percentage = 0.01
fig = plt.figure(figsize=(12, 12), dpi=300, facecolor="white")

# Calculate the Mean Squared Error (MSE) for all data points
train_mse = mean_squared_error(train_y, train_pred_y)
val_mse = mean_squared_error(val_y, val_pred_y)
test_mse = mean_squared_error(test_y, test_pred_y)

# Round the MSE values to three digits
train_mse = round(train_mse, 3)
val_mse = round(val_mse, 3)
test_mse = round(test_mse, 3)

# Calculate the sample size as a percentage of the total points in each set
train_sample_size = int(len(train_y) * sample_percentage)
val_sample_size = int(len(val_y) * sample_percentage)
test_sample_size = int(len(test_y) * sample_percentage)

# Sample a subset of data points for plotting
train_sample_indices = np.random.choice(len(train_y), size=train_sample_size, replace=False)
val_sample_indices = np.random.choice(len(val_y), size=val_sample_size, replace=False)
test_sample_indices = np.random.choice(len(test_y), size=test_sample_size, replace=False)

train_sample_y = train_y[train_sample_indices]
train_sample_pred_y = train_pred_y[train_sample_indices]
val_sample_y = val_y[val_sample_indices]
val_sample_pred_y = val_pred_y[val_sample_indices]
test_sample_y = test_y[test_sample_indices]
test_sample_pred_y = test_pred_y[test_sample_indices]


# Calculate the CDF for each set of data points
train_sample_y_sorted = np.sort(train_sample_y)
train_sample_pred_y_sorted = np.sort(train_sample_pred_y)
train_y_cdf = np.arange(1, train_sample_size + 1) / train_sample_size * 100
train_pred_y_cdf = np.arange(1, train_sample_size + 1) / train_sample_size * 100
val_sample_y_sorted = np.sort(val_sample_y)
val_sample_pred_y_sorted = np.sort(val_sample_pred_y)
val_y_cdf = np.arange(1, val_sample_size + 1) / val_sample_size * 100
val_pred_y_cdf = np.arange(1, val_sample_size + 1) / val_sample_size * 100
test_sample_y_sorted = np.sort(test_sample_y)
test_sample_pred_y_sorted = np.sort(test_sample_pred_y)
test_y_cdf = np.arange(1, test_sample_size + 1) / test_sample_size * 100
test_pred_y_cdf = np.arange(1, test_sample_size + 1) / test_sample_size * 100

# Set up the grid layout
axes1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # This will span 3 columns
axes2 = plt.subplot2grid((3, 3), (1, 0))  # Scatterplot for training set
axes3 = plt.subplot2grid((3, 3), (1, 1))  # Scatterplot for validation set
axes4 = plt.subplot2grid((3, 3), (1, 2))  # Scatterplot for test set
axes5 = plt.subplot2grid((3, 3), (2, 0))  # CDF for training set
axes6 = plt.subplot2grid((3, 3), (2, 1))  # CDF for validation set
axes7 = plt.subplot2grid((3, 3), (2, 2))  # CDF for test set

# Plot for the top 5 data comparison
# Define the colors for the lines
colors = ['blue', 'green', 'red', 'purple', 'orange']
line_colors = itertools.cycle(colors)

# Iterate over all 36 rows of the DataFrame
for index, row in top_36_data_df.iterrows():
    # Extract LR, BS, LRF, and IFN values
    LR = row['LR']
    BS = int(row['BS'])
    IFN = int(row['IFN'])

    # Generate the training_unique_name
    loss = "mse"
    min_LR = 0.00000000000001
    lr_patience = 2
    patience = 8
    epochs = 64
    val_split = 0.1
    training_unique_name = f"{loss}_{IFN}_{LR}_{min_LR}_{LRF}_{lr_patience}_{int(BS)}_{patience}_{val_split}_{epochs}"

    file_path = HPT_PATH + training_unique_name + ".csv"
    data = pd.read_csv(file_path)

    if index > 4:
        axes1.plot(data['mse'].index + 1, data['mse'], label='', linewidth=0.8, color='gray', alpha=0.25, zorder=1)
        axes1.plot(data['val_mse'].index + 1, data['val_mse'], label='', linestyle='dashed', linewidth=0.8, color='gray', zorder=1, alpha=0.25)
    else:
        line_color = next(line_colors)
        axes1.plot(data['mse'].index + 1, data['mse'], label=f'LR={LR}, BS={BS}, LRF={LRF}, IFN={IFN}', linewidth=0.8, color=line_color)
        axes1.plot(data['val_mse'].index + 1, data['val_mse'], label='', linestyle='dashed', linewidth=0.8, color=line_color)
        
def plot_scatterplot(ax, sample_y, sample_pred_y, title, sample_size, total_size, mse):
    ax.scatter(sample_y, sample_pred_y, s=10, alpha=0.5)
    ax.set_xlabel("Actual Mismatch (mm/hr)")
    ax.set_ylabel("Predicted Mismatch (mm/hr)")
    ax.set_title(f"{title}\nSampled Data Points: {sample_size} out of\n {total_size}\nMSE: {mse:.3f}")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.grid(True)
    
axes1.axvline(x=best_epoch, ymax=0.5, color='blue', linestyle='dotted', linewidth=0.8, label='Early Stop Epoch')
axes1.set_title('Loss Comparison - Top 5', fontsize=16)
axes1.set_xlabel('Epoch', fontsize=14)
axes1.set_ylabel('MSE - mm/hr', fontsize=14)
axes1.tick_params(axis='x', labelsize=12)
axes1.tick_params(axis='y', labelsize=12)
axes1.legend(fontsize=12, frameon=False)
axes1.grid(True, linestyle='--', alpha=0.5)
axes1.set_ylim(0.15, 0.40)
axes1.set_xlim(1, 25)

# Update plot_scatterplot calls with total_size argument
plot_scatterplot(axes2, train_sample_y, train_sample_pred_y, "Scatterplot: Training Set", train_sample_size, len(train_y), train_mse)
plot_scatterplot(axes3, val_sample_y, val_sample_pred_y, "Scatterplot: Validation Set", val_sample_size, len(val_y), val_mse)
plot_scatterplot(axes4, test_sample_y, test_sample_pred_y, "Scatterplot: Test Set", test_sample_size, len(test_y), test_mse)

def plot_cdf(ax, data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data) * 100
    ax.plot(sorted_data, cdf, label=label, linewidth=2)
    ax.set_xlabel('Mismatch (mm/hr)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-12, 12)  # Set the x-axis limits to -12 and 12
    
# CDF for training set
plot_cdf(axes5, train_sample_y, 'Actual')
plot_cdf(axes5, train_sample_pred_y, 'Predicted')
axes5.set_title('CDF: Training Set', fontsize=14)

# CDF for validation set
plot_cdf(axes6, val_sample_y, 'Actual')
plot_cdf(axes6, val_sample_pred_y, 'Predicted')
axes6.set_title('CDF: Validation Set', fontsize=14)

# CDF for test set
plot_cdf(axes7, test_sample_y, 'Actual')
plot_cdf(axes7, test_sample_pred_y, 'Predicted')
axes7.set_title('CDF: Test Set', fontsize=14)

# Adjust the layout and save the plot
plt.tight_layout()
plt.savefig(DUMP_PLOT + '/HPT_Training_Results_batch.png', dpi=300)
plt.show()