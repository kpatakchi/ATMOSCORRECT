from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Hyperparameter Tuning Parameters")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--bs", type=int, required=True, help="Batch size")
parser.add_argument("--lr_factor", type=float, required=True, help="Learning rate factor")
parser.add_argument("--filters", type=int, required=True, help="Number of filters")
parser.add_argument("--date_start", type=str, required=True, help="Start date")
parser.add_argument("--date_end", type=str, required=True, help="End date")
args = parser.parse_args()

LR=args.lr
BS=args.bs
lr_factor=args.lr_factor
Filters=args.filters
date_start=args.date_start
date_end=args.date_end

# Define the data specifications:
model_data = ["HRES"]
reference_data = ["HSAF"]
task_name = "spatiotemporal"
mm = "MM"  # or DM
#date_start = "2020-07-01T13"
#date_end = "2020-07-26T23"
min_delta_or_lr=0.00000000000001 #just to avoid any limitations

variable = "pr"
mask_type = "no_na"
laginensemble = False

# Define the following for network configs:
loss = "mse"
min_LR = min_delta_or_lr
lr_patience = 2
patience = 8
epochs = 64
val_split = 0.1
n_channels = 7
xpixels = 128
ypixels = 256

filename = Func_Train.data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)
data_unique_name = filename[:-4]
print(data_unique_name)

training_unique_name = Func_Train.generate_training_unique_name(loss, Filters, LR, min_LR, lr_factor, 
                                                    lr_patience, BS, patience, val_split, 
                                                    epochs)

# Create the production data (if doesn't exist)
Func_Train.prepare_produce(PPROJECT_DIR, PRODUCE_FILES, ATMOS_DATA, filename, model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble)

# load the production data
print("Loading production data...")
produce_files=np.load(PRODUCE_FILES+"/"+"produce_for_"+filename)
produce_x=produce_files["canvas_x"]

# load the model and weights
model = Func_Train.UNET_ATT(xpixels, ypixels, n_channels, Filters)
model_path = PSCRATCH_DIR + "/HPT_v1/" + training_unique_name + ".h5"
model.load_weights(model_path)

# produce 
Y_PRED = model.predict(produce_x, verbose=1)
Y_PRED=Y_PRED[..., 0]

# Save in PREDICT_FILES
np.savez(PREDICT_FILES + "/predicted_for_" + data_unique_name + "_" +training_unique_name, Y_PRED=Y_PRED)
print("saved")

# Save in PREDICT_FILES
Func_Train.de_prepare_produce(Y_PRED, PREDICT_FILES, ATMOS_DATA, filename, 
                              model_data, date_start, date_end, variable, 
                              training_unique_name, onedelay=True)