from py_env_train import *
import argparse

# define the parameters
parser = argparse.ArgumentParser(description="Hyperparameter Tuning Parameters")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--bs", type=int, required=True, help="Batch size")
parser.add_argument("--lr_factor", type=float, required=True, help="Learning rate factor")
parser.add_argument("--filters", type=int, required=True, help="Number of filters")
args = parser.parse_args()

# Define the data specifications:
model_data = ["HRES"]
reference_data = ["HSAF"]
task_name = "spatiotemporal"
mm = "MM"  # or DM
date_start = "2020-07-01T13"
date_end = "2023-04-26T23"
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

# Create the training data (if doesn't exist)
data_avail = Func_Train.prepare_train(PPROJECT_DIR, TRAIN_FILES, ATMOS_DATA, filename, 
                         model_data, reference_data, task_name, mm, date_start,
                         date_end, variable, mask_type, laginensemble, val_split)

# load the training data
print("Loading training data...")
train_files = np.load(TRAIN_FILES + "/" + filename)

train_x = train_files["train_x"]
train_y = train_files["train_y"]
train_m = train_files["train_m"]
val_x = train_files["val_x"]
val_y = train_files["val_y"]
val_m = train_files["val_m"]

LR=args.lr
BS=args.bs
lr_factor=args.lr_factor
Filters=args.filters

training_unique_name = Func_Train.generate_training_unique_name(loss, Filters, LR, min_LR, lr_factor, 
                                                    lr_patience, BS, patience, val_split, 
                                                    epochs)
print(training_unique_name)

import tensorflow as tf

def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    zero_precip_indices = tf.where(tf.equal(y_true, 0))
    zero_precip_pred = tf.gather_nd(y_pred, zero_precip_indices)
    penalty = tf.reduce_mean(tf.square(zero_precip_pred))
    alpha = 5
    combined_loss = mse_loss + alpha * penalty
    return combined_loss

model = Func_Train.UNET_ATT(xpixels, ypixels, n_channels, Filters)

# Define optimizer and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, name='Adam')
#model.compile(optimizer=optimizer, loss=loss, weighted_metrics=['mse'], sample_weight_mode='temporal')
model.compile(optimizer=optimizer, loss=loss, weighted_metrics=["mse"])

# Define the model checkpoint and early stopping callbacks
model_path = PSCRATCH_DIR + '/BEST_DL/' + training_unique_name + '.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=2, save_best_only=True, monitor='val_loss')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir=PSCRATCH_DIR + '/BEST_DL/' + training_unique_name)]

# Define the ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_LR, min_delta=min_delta_or_lr)

print("Training the model...")

# Train the model using train_x, train_y, train_m and val_x, val_y, val_m
results = model.fit(train_x, train_y, validation_data=(val_x, val_y, val_m),
                    batch_size=BS, epochs=epochs, verbose=2,
                    callbacks=[callbacks, checkpointer, reduce_lr], sample_weight=train_m,
                   shuffle=True)

# Save and plot the results
print("Saving and plotting the results...")
RESULTS_DF = pd.DataFrame(results.history)
RESULTS_DF.to_csv(PSCRATCH_DIR + "/BEST_DL/" + training_unique_name + ".csv")