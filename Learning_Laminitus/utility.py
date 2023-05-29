import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf 

healthy = ['10', '13', '25', '06', '12', '19', '22', '23']  # Healthy ID's

path_to_interpolated_data = '/Users/rileyoest/VS_Code/Learning_Laminitus/Interpolated' # Current path to Directory with Interpolated Data 

#                    Acquiring DataFrame                 #

def acquire_all_data():
    frames = []
    for shoe in os.listdir(path_to_interpolated_data):
        for horse in os.listdir(path_to_interpolated_data + "/" + shoe + "/"):
            # horse looks like "Hoof03_EB"
            isLaminitic = 1
            if horse[-5:-3] in healthy:
                isLaminitic = 0
            for trial in os.listdir(path_to_interpolated_data + "/" + shoe + "/" + horse + "/"):
                # trial looks like "Hoof03_EB_02.csv"
                data = pd.read_csv(
                    path_to_interpolated_data + "/" + shoe + "/" + horse + "/" + trial)
                data['isLaminitic'] = isLaminitic
                data['Shoe'] = horse[-2:]
                frames.append(data)
    df = pd.concat(frames)
    df = df.dropna()
    df['Trial'] = (df['Time'] == 0).cumsum()
    df = df.reindex(columns=['Trial'] + list(df.columns[:-1]))
    df["Shoe"] = df["Shoe"].replace({"OH": 1, "US": 2, "HB": 3, "EB": 4})
    return df


"""
This function returns the dataframe of specified variables. It also adds a column "Trial"
to keep track of which trial each row of data belongs to. 

Inputs : shoe_type, health_type

shoe_type's available:
=====================
Standards = [1]
Heartbars = [2]
Eggbars = [3]
Unshods = [4]

health_type's available:
=====================
Not Healthy = [0]
Healthy = [1]

Outputs : dataframe of trials with specified shoe_type and health_type
"""


def acquire_data(shoe_type, health_type):
    if health_type == 1:
        health_type = False  # Horse is healthy
    else:
        health_type = True  # Horse is not healthy
    frames = []
    for shoe in os.listdir(path_to_interpolated_data):
        # shoe looks like "Hoof_EB"
        if (shoe_type == 1) and ("OH" in shoe):
            for horse in os.listdir(path_to_interpolated_data + "/" + shoe + "/"):
                # horse looks like "Hoof03_EB"
                isLaminitic = True
                if horse[-5:-3] in healthy:
                    isLaminitic = False
                if health_type == isLaminitic:
                    for trial in os.listdir(path_to_interpolated_data + "/" + shoe + "/" + horse + "/"):
                        # trial looks like "Hoof03_EB_02.csv"
                        data = pd.read_csv(
                            path_to_interpolated_data + "/" + shoe + "/" + horse + "/" + trial)
                        frames.append(data)
        elif shoe_type == 2 and "HB" in shoe:
            for horse in os.listdir(path_to_interpolated_data + "/" + shoe + "/"):
                # horse looks like "Hoof03_EB"
                isLaminitic = True
                if horse[-5:-3] in healthy:
                    isLaminitic = False
                if health_type == isLaminitic:
                    for trial in os.listdir(path_to_interpolated_data + "/" + shoe + "/" + horse + "/"):
                        # trial looks like "Hoof03_EB_02.csv"
                        data = pd.read_csv(
                            path_to_interpolated_data + "/" + shoe + "/" + horse + "/" + trial)
                        frames.append(data)
        elif shoe_type == 3 and "EB" in shoe:
            for horse in os.listdir(path_to_interpolated_data + "/" + shoe + "/"):
                # horse looks like "Hoof03_EB"
                isLaminitic = True
                if horse[-5:-3] in healthy:
                    isLaminitic = False
                if health_type == isLaminitic:
                    for trial in os.listdir(path_to_interpolated_data + "/" + shoe + "/" + horse + "/"):
                        # trial looks like "Hoof03_EB_02.csv"
                        data = pd.read_csv(
                            path_to_interpolated_data + "/" + shoe + "/" + horse + "/" + trial)
                        frames.append(data)
        elif shoe_type == 4 and "US" in shoe:
            for horse in os.listdir(path_to_interpolated_data + "/" + shoe + "/"):
                # horse looks like "Hoof03_EB"
                isLaminitic = True
                if horse[-5:-3] in healthy:
                    isLaminitic = False
                if health_type == isLaminitic:
                    for trial in os.listdir(path_to_interpolated_data + "/" + shoe + "/" + horse + "/"):
                        # trial looks like "Hoof03_EB_02.csv"
                        data = pd.read_csv(
                            path_to_interpolated_data + "/" + shoe + "/" + horse + "/" + trial)
                        frames.append(data)
    df = pd.concat(frames)
    df = df.dropna()
    df['Trial'] = (df['Time'] == 0).cumsum()
    df = df.reindex(columns=['Trial'] + list(df.columns[:-1]))
    if shoe_type == 1 and health_type == True:
        print("="*55, 'Healthy Standards', "="*55)
    if shoe_type == 1 and health_type == False:
        print("="*55, 'Laminitic Standards', "="*55)
    if shoe_type == 2 and health_type == True:
        print("="*55, 'Healthy Heartbars', "="*55)
    if shoe_type == 2 and health_type == False:
        print("="*55, 'Laminitic Heartbars', "="*55)
    if shoe_type == 3 and health_type == True:
        print("="*55, 'Healthy Eggbars', "="*55)
    if shoe_type == 3 and health_type == False:
        print("="*55, 'Laminitic Eggbars', "="*55)
    if shoe_type == 4 and health_type == True:
        print("="*55, 'Healthy Unshods', "="*55)
    if shoe_type == 4 and health_type == False:
        print("="*55, 'Laminitic Unshods', "="*55)
    return df

# ------------------------------------------------------ #



#                       Translation                      #
"""
translate_data()

input: dataframe of all trials
outputs: dataframe of all trials with fixed starting points for each coordinate
"""
def translate_data(df):
    df = df.copy()
    trials = df['Trial'].unique()

    for trial in trials:
        data = df['Trial'] == trial
        v0 = df.loc[data & (df['Time'] == 0), [
            'DW_x',	'DW_y',	'DW_z',	'SM_x',	'SM_y',	'SM_z',	'CB_x',	'CB_y',	'CB_z',	'P3_x',	'P3_y',	'P3_z']].values

        df.loc[data, [
            'DW_x',	'DW_y',	'DW_z',	'SM_x',	'SM_y',	'SM_z',	'CB_x',	'CB_y',	'CB_z',	'P3_x',	'P3_y',	'P3_z']] -= v0
    
    return df

# ------------------------------------------------------ #


#                Equalizing Trial Lengths                #
def equalize_trial_lengths(df):
    n = (df['Time'] == 0).sum()
    trial_lengths = {}

    for i in range(1, n+1):
        trial = df[df['Trial'] == i]
        trial_lengths[i] = len(trial)

    equalized_df = pd.DataFrame()

    for trial_number in df['Trial'].unique():
        trial = df[df['Trial'] == trial_number].copy()
        trial_length = len(trial)

        if trial_length > min(trial_lengths.values()):
            trial = trial.sort_values('Time').head(min(trial_lengths.values()))

        equalized_df = pd.concat([equalized_df, trial])

    equalized_df.reset_index(drop=True, inplace=True)
    return equalized_df

# ------------------------------------------------------ #

#                            Normalization               #

def normalize_coords(data):
    df = data.copy()
    n = (df['Time'] == 0).sum()
    for i in range(1, n+1):
        trial = df[df['Trial'] == i]
        coords = ['DW_x', 'DW_y', 'DW_z', 'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']
        for coord in coords:
            min_val = trial[coord].min()
            max_val = trial[coord].max()
            df.loc[df['Trial'] == i, coord] = (df[df['Trial'] == i][coord] - min_val) / (max_val - min_val)

    return df
# ------------------------------------------------------ #


#                   Plotting Coordinates                 #
"""
Inputs : trial (number), df (dataframe)
Outputs : plt plot 
"""


def plot_coords(trial, df):
    print(f"Plot of trial {trial} coordinates :")
    rows = 4
    cols = 3
    _, axis = plt.subplots(rows, cols, figsize=(25, 25))
    coords = ['DW_x', 'DW_y', 'DW_z',	'SM_x',	'SM_y',	'SM_z',
              'CB_x',	'CB_y',	'CB_z',	'P3_x',	'P3_y',	'P3_z']
    i = 0
    while i < rows*cols:
        row = i // cols
        col = i % cols
        axis[row, col].plot(df[df['Trial'] == trial]['Time'],
                            df[df['Trial'] == trial][coords[i]])
        axis[row, col].set_xlabel("Time")
        axis[row, col].set_ylabel(coords[i])
        axis[row, col].set_title(coords[i])
        i += 1
    plt.show()


"""
Inputs : trial (number), coord ("DW_x"), df (dataframe)
Outputs : plt plot
"""


def plot_coord(trial, coord, df):
    plt.plot(df[df['Trial'] == trial]['Time'], df[df['Trial'] == trial][coord])
    plt.title(f'Trial {trial} {coord} along time')
    plt.xlabel("Time")
    plt.ylabel(coord)
    plt.show()

# ------------------------------------------------------ #

#                   Train/Test Split                     #

"""
This function takes the given dataframe and outputs the train test split
in the following layered format : 

X layer = 'Time', 'Trial', 'DW_x', 'SM_x', 'CB_x',
Y layer = 'Time', 'Trial', 'DW_y', 'SM_y', 'CB_y',
Z layer = 'Time', 'Trial', 'DW_z', 'SM_z', 'CB_z', 
y layer = 'Time', 'Trial', 'P3_x', 'P3_y', and 'P3_z'

Each of these layers will have train and test dataframes. Each training dataframe
will have train_size % of the trials and each testing dataframe will have 1 - train_size % of the trials

Inputs : dataframe, train_size
Ouputs : X_layer_train, X_layer_test, Y_layer_train, Y_layer_test, Z_layer_train, Z_layer_test, y_train, y_test
"""


def train_test_split_layers(df, train_size = 0.7):
    n = (df['Time'] == 0).sum()
    train = int(train_size * n)
    X_layer_train = []
    X_layer_test = []
    Y_layer_train = []
    Y_layer_test = []
    Z_layer_train = []
    Z_layer_test = []
    y_train = []
    y_test = []
    for i in range(n):
        trial_data = df[df['Trial'] == i + 1][['Time', 'Trial', 'Shoe', 'isLaminitic', 'DW_x', 'DW_y', 'DW_z',
                                               'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z']]
        # if i <= train:
        #     X_layer_train.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_x', 'SM_x', 'CB_x']])
        #     Y_layer_train.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_y', 'SM_y', 'CB_y']])
        #     Z_layer_train.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_z', 'SM_z', 'CB_z']])
        #     y_train.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'P3_x', 'P3_y', 'P3_z']])
        # else:
        #     X_layer_test.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_x', 'SM_x', 'CB_x']])
        #     Y_layer_test.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_y', 'SM_y', 'CB_y']])
        #     Z_layer_test.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'DW_z', 'SM_z', 'CB_z']])
        #     y_test.append(
        #         trial_data[['Time', 'isLaminitic', 'Shoe', 'P3_x',  'P3_y', 'P3_z']])
            

        if i <= train:
            X_layer_train.append(
                trial_data[['Time', 'DW_x', 'SM_x', 'CB_x']])
            Y_layer_train.append(
                trial_data[['Time', 'DW_y', 'SM_y', 'CB_y']])
            Z_layer_train.append(
                trial_data[['Time', 'DW_z', 'SM_z', 'CB_z']])
            y_train.append(
                trial_data[['Time', 'P3_x', 'P3_y', 'P3_z']])
        else:
            X_layer_test.append(
                trial_data[['Time', 'DW_x', 'SM_x', 'CB_x']])
            Y_layer_test.append(
                trial_data[['Time', 'DW_y', 'SM_y', 'CB_y']])
            Z_layer_test.append(
                trial_data[['Time', 'DW_z', 'SM_z', 'CB_z']])
            y_test.append(
                trial_data[['Time', 'P3_x',  'P3_y', 'P3_z']])
            

    X_layer_train = pd.concat(X_layer_train, axis=0)
    X_layer_test = pd.concat(X_layer_test, axis=0)
    Y_layer_train = pd.concat(Y_layer_train, axis=0)
    Y_layer_test = pd.concat(Y_layer_test, axis=0)
    Z_layer_train = pd.concat(Z_layer_train, axis=0)
    Z_layer_test = pd.concat(Z_layer_test, axis=0)
    y_train = pd.concat(y_train, axis=0)
    y_test = pd.concat(y_test, axis=0)
    return X_layer_train, X_layer_test, Y_layer_train, Y_layer_test, Z_layer_train, Z_layer_test, y_train, y_test
# ------------------------------------------------------ #


#                       Build Model                      #
def build_model():
    X_input = tf.keras.layers.Input(shape = (4, ), name = 'X_input')
    Y_input = tf.keras.layers.Input(shape = (4, ), name = 'Y_input')
    Z_input = tf.keras.layers.Input(shape = (4, ), name = 'Z_input')

    dense = tf.keras.layers.Dense(units = 64, activation = 'relu', name= 'shared_dense_layer')

    shared_X = dense(X_input)
    shared_Y = dense(Y_input)
    shared_Z = dense(Z_input)

    merged = tf.keras.layers.concatenate([shared_X, shared_Y, shared_Z])

    X = tf.keras.layers.Dense(64, activation = 'relu')(merged)
    X = tf.keras.layers.Dense(64, activation = 'relu')(X)

    outputs = tf.keras.layers.Dense(4)(X)

    model = tf.keras.Model(inputs = [X_input, Y_input, Z_input], outputs = outputs)

    return model
# ------------------------------------------------------ #

#                   Plot Results                         #

"""
plot_results() 
inputs: y_true and y_pred
* y_true comes from train_test_split_layers()
* y_pred comes from the model's prediction

outputs: 3x2 graph of true and predicted P3 coordinates
"""

def plot_results(y_true, y_pred):
    time = []
    P3_x = []
    P3_y = []
    P3_z = []
    for t, x, y, z in y_pred:
        time.append(t)
        P3_x.append(x)
        P3_y.append(y)
        P3_z.append(z)

    print('Plotting true coordinates vs predicted coordinates')
    rows = 3
    cols = 2
    _, ax = plt.subplots(rows, cols,  figsize=(20, 25))
    coords = ['P3_x', 'P3_x Predicted', 'P3_y', 'P3_y Predicted', 'P3_z', 'P3_z Predicted']
    i = 0
    while i < rows*cols:
        row = i // cols
        col = i % cols
        if (i % 2 == 0):
            ax[row, col].plot(y_true['Time'], y_true[coords[i]])
            ax[row, col].set_xlabel('Time')
            ax[row, col].set_ylabel(f'{coords[i]} True')
            ax[row, col].set_title(f'{coords[i]} True')
        else:
            if i == 1:
                ax[row, col].plot(time, P3_x)
            elif i == 3:
                ax[row, col].plot(time, P3_y)
            else:
                ax[row, col].plot(time, P3_z)
            ax[row, col].set_xlabel('Time')
            ax[row, col].set_ylabel(coords[i])
            ax[row, col].set_title(f'{coords[i]} Predicted')
        i += 1
        
# ------------------------------------------------------ #






##########################################################

#                       (For Reference)                  #

def print_trial_lengths(df):
    n = (df['Time'] == 0).sum()
    trial_lengths = {}

    for i in range(n):
        trial = df[df['Trial'] == i + 1]
        last_time_value = trial['Time'].values[-1]  # Extract the last value of the 'Time' column
        
        #trial_lengths[i + 1] = last_time_value
        trial_lengths[i + 1] = len(trial)

    print(trial_lengths)


    

##########################################################
#                      Interpolation                     #
"""
find_zero_spikes()

inputs: dataframe of one horse trial, and a coordinate in the corresponding trial (ie: "DW_x")
outputs: The times where that coordinate is found to be zero
"""


def find_zero_spikes(df, coord):
    abnormal_spikes = []
    abnormal_spikes = [df['Time'][x]
                       for x in range(len(df)) if (df[coord][x] == 0)]
    return abnormal_spikes


"""
interpolate_spikes()

inputs: dataframe of one horse trial, list of times where a certain coordinate is found to be zero, and the name of that coordinate
outputs: the dataframe of the horse trial with interpolated times where zeros occurred.
"""


def interpolate_spikes(df, spikes, coord):
    df = df.copy()
    mean = df[coord].mean()
    std = df[coord].std()
    for i in range(len(df)):
        if (df['Time'][i] in spikes) and (df['Time'][i] != 0) and (df['Time'][i] != df.tail(1)['Time'].values[0]):
            left_index = i - 1
            right_index = i + 1
            while (left_index >= 0) and ((df[coord][left_index] == 0) or (df[coord][left_index] < mean - 1 * std)):
                left_index -= 1
            while (right_index < len(df)) and ((df[coord][right_index] == 0) or (df[coord][right_index] < mean - 1 * std)):
                right_index += 1
            if (left_index >= 0) and (right_index < len(df)):
                left_time = df['Time'][left_index]
                left_coord = df[coord][left_index]
                right_time = df['Time'][right_index]
                right_coord = df[coord][right_index]
                interp_times = df['Time'][(df['Time'] >= left_time) & (
                    df['Time'] <= right_time)]
                interp_coords = np.interp(interp_times, [left_time, right_time], [
                                          left_coord, right_coord])
                df.loc[(df['Time'] >= left_time) & (
                    df['Time'] <= right_time), coord] = interp_coords
            elif left_index >= 0:
                df.at[i, coord] = df[coord][left_index]

            elif right_index < len(df):
                df.at[i, coord] = df[coord][right_index]
    return df


"""
This function loops through the "cleaned data" which is the training data, trimmed down
to only include timepoints that are valid in the experiment. It manipulates creates readable 
column titles for each x, y, z coordinate and also includes columns for time values,  boolean values 'laminitic',
and shoe type. This function utilizes find_zero_spikes() and interpolate_spikes() to interpolate data for each column (coordinate)
within each dataset. Finally, after all interpolation, the function saves a dataset to a new directory.
"""


def interpolate_data():
    path_to_cleaned_data = "/Users/rileyoest/VS_Code/PythonProjects/Hoof_Learning/Cleaned_Data" # Directory doesn't exist anymore
    count = 1
    frame = pd.DataFrame(columns=['Laminitic', 'Shoe', 'Time', 'DW_x', 'DW_y',
                                  'DW_z', 'SM_x', 'SM_y', 'SM_z', 'CB_x', 'CB_y', 'CB_z', 'P3_x', 'P3_y', 'P3_z'])
    for shoe in os.listdir(path_to_cleaned_data):
        # shoe looks like "Hoof_EB"
        for horse in os.listdir(path_to_cleaned_data + "/" + shoe + "/"):
            # horse looks like "Hoof03_EB"
            isLaminitic = True
            if horse[-5:-3] in healthy:
                isLaminitic = False
            for trial in os.listdir(path_to_cleaned_data + "/" + shoe + "/" + horse + "/"):
                # trial looks like "Hoof03_EB_02.csv"
                data = pd.read_csv(path_to_cleaned_data +
                                   "/" + shoe + "/" + horse + "/" + trial)
                # Renaming column titles
                data = data.rename(columns={'Marker 09.X': 'DW_x', 'Marker 09.Y': 'DW_y', 'Marker 09.Z': 'DW_z', 'Marker 12.X': 'SM_x', 'Marker 12.Y': 'SM_y', 'Marker 12.Z': 'SM_z',
                                            'Marker 10.X': 'CB_x', 'Marker 10.Y': 'CB_y', 'Marker 10.Z': 'CB_z', 'Marker 11.X': 'P3_x', 'Marker 11.Y': 'P3_y', 'Marker 11.Z': 'P3_z'})
                # Drop P3 Medial Coordinates
                # Medial cooridnates already taken out in 12-21
                data.drop(columns=data.columns[-3:], axis=1, inplace=True)
                time = np.array(data.iloc[:, 0])
                # Interpolation
                for coord in data.columns[1:]:
                    zeros = find_zero_spikes(data, coord)
                    bad_times = list(set(zeros).symmetric_difference(time))
                    data = interpolate_spikes(data, bad_times, coord)
                l = [isLaminitic, horse[-2:], list(time), data.iloc[:, 1], data.iloc[:, 2], data.iloc[:, 3], data.iloc[:, 4], data.iloc[:, 5],
                     data.iloc[:, 6], data.iloc[:, 7], data.iloc[:, 8], data.iloc[:, 9], data.iloc[:, 10], data.iloc[:, 11], data.iloc[:, 12]]
                frame.loc[count] = l
                count += 1
                filename = f"/Users/rileyoest/VS_Code/PythonProjects/Hoof_Learning/Interpolated/{shoe}/{horse}/{trial}"
                data.to_csv(filename, index=False)
# ------------------------------------------------------ #
