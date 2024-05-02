#!/usr/bin/env python

from py_env_train import *
import argparse

# ----- Module level variables -----
me = sys.modules[__name__]
me.batch_size = ""
me.network = ""
# ----------------------------------

def process_script_args():
    """
    Validates command line arguments. Returns True if all arguments are valid.
    """

    helpText = '''More information:
    https://icg4geo.icg.kfa-juelich.de/SoftwareTools/prepro_clm5_mklandfiles
    '''

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=helpText,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-n',
        "--ntasks",
        default=me.numRegridTasks,
        type=int,
        help="(Experimental) Number of regridding tasks. Default value is " +
        str(me.numRegridTasks))
    parser.add_argument(
        "-t",
        "--test",
        action='store_true',
        help="Test only; do not submit batch script to queue.")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__)
    try:
        args = parser.parse_args()
    except BaseException:
        print("")
        parser.print_help()
        return False

    me.batch_size = args.batch_size

    return True

def main():
        # Define the following:
    model_data = ["HRES"] # TSMP must come first for calculating the mismatch correctly in ensembles!!!
    reference_data = ["HSAF"]
    task_name = "spatiotemporal"
    mm = "MM"  # or DM
    date_start="2020-10-01"
    date_end="2021-09-30"
    variable="pr"
    mask_type="no_na"
    laginensemble=False

    # The following is defined automatically:
    n_ensembles = len(model_data)
    n_channels = Func_Train.calculate_channels(n_ensembles, task_name, laginensemble=laginensemble)
    if reference_data == ["COSMO_REA6"]:
        canvas_size = (400, 400) 
        topo_dir='/p/project/deepacf/kiste/patakchiyousefi1/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
        trim=True
        daily=True
    if reference_data == ["HSAF"]:
        topo_dir='/p/project/deepacf/kiste/patakchiyousefi1/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
        canvas_size = (128, 256)
        trim=False
        daily=False
    data_unique_name = f"train_data{'_daily' if daily else '_hourly'}.{variable}.{model_data}.{reference_data}.{mm}.{n_channels}.{'laginensemble' if laginensemble else ''}.{task_name}.{'.'.join(map(str, canvas_size))}.{date_start}.{date_end}.{mask_type}"
    filename = f"{data_unique_name}.npz"

    # load the data and define the training configurations:
    train_files=np.load(TRAIN_FILES+"/"+filename)
    xpixels=train_files["canvas_x"].shape[1]
    ypixels=train_files["canvas_x"].shape[2]

    # Define the following for network configs (the fixed hyperparameters)
    loss="mse"
    Filters=32
    patience=8
    epochs=64
    val_split=0.25

    learning_rates = [0.001, 0.05, 0.01, 0.05, 0.1]
    batch_sizes = [1, 2, 5, 10, 24, 48]

    # Define the variable hyperparameters (LR and BS):
    for LR in learning_rates:
        for BS in batch_sizes:
            training_unique_name = data_unique_name+"."+loss+"."+str(Filters)+"."+str(LR)+"."+str(BS)+"."+str(patience)+"."+str(val_split)+"."+str(epochs)
            print("Training: BS: ", str(BS), "LR: ", str(LR))

            model = Func_Train.UNET(xpixels, ypixels, n_channels, Filters)
            optimizer = tf.keras.optimizers.Adam(learning_rate=LR, name='Adam')
            model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

            model_path = '/p/project/deepacf/kiste/patakchiyousefi1/AI MODELS/00-UNET/'+training_unique_name+'.h5'
            checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=2, save_best_only=True, monitor='val_loss')
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
                         tf.keras.callbacks.TensorBoard(log_dir='/p/project/deepacf/kiste/patakchiyousefi1/AI MODELS/00-UNET/'+training_unique_name)]

            results = model.fit(train_files["canvas_x"], train_files["canvas_y"], 
                                validation_split=val_split, 
                                batch_size=BS, 
                                epochs=epochs, 
                                verbose=1, 
                                callbacks=[callbacks, checkpointer],
                                sample_weight=train_files["canvas_m"], 
                                shuffle=False)

            results_df = pd.DataFrame(results.history)
            results_df.to_csv(DUMP_RESULTS+"/"+training_unique_name+".csv")

# Main program entrypoint
if __name__ == "__main__":
    process_script_args()
    print ("hi")
    #main()
