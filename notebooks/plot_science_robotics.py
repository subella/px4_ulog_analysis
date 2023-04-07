# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: trt_venv
#     language: python
#     name: envname
# ---

# +
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
import bisect
import subprocess
import numpy as np
import functools
import pyquaternion as pyq

import sys
import pandas as pd
from bagpy import bagreader
import os
# -

sns.set()

# +
MESSAGES = [
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "position_setpoint_triplet",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "vehicle_rates_setpoint",
]
MESSAGE_ARGS = ",".join(MESSAGES)

NAME_TO_MESSAGES = {
    "position": "vehicle_local_position_0",
    "position_sp": "vehicle_local_position_setpoint_0",
    "attitude": "vehicle_attitude_0",
    "attitude_sp": "vehicle_attitude_setpoint_0",
    "commanded": "position_setpoint_triplet_0",
    "attitude_output": "vehicle_rates_setpoint_0",
}


# +
def get_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return list(log_dir.glob("*.ulg"))


def get_array_values(df, name, shape):
    """Extract values into a numpy array."""
    total_size = functools.reduce(lambda x, y: x * y, shape, 1)
    columns = ["{}[{}]".format(name, i) for i in range(total_size)]
    return np.array(
        [
            np.array([df.iloc[i][c] for c in columns]).reshape(shape)
            for i in range(len(df))
        ]
    )


def get_field_values(df, fields, shape):
    """Extract message fields into a numpy array."""
    total_size = functools.reduce(lambda x, y: x * y, shape, 1)
    return np.array(
        [
            np.array([df.iloc[i][c] for c in fields]).reshape(shape)
            for i in range(len(df))
        ]
    )


def get_df_with_time(data_frames, name, first_time, timescale=1.0e-6):
    """Add a time column to the dataframe."""
    print(data_frames)
    to_return = data_frames[name]
    to_return["times"] = (to_return["timestamp"] - first_time) * 1.0e-6
    return to_return


def get_dataframes(result_dir, file_to_use):
    """Get dataframes from the result directory."""
    message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))

    if len(message_csvs) == 0:
        subprocess.call(
            ["ulog2csv", str(file_to_use), "-o", str(result_dir), "-m", MESSAGE_ARGS]
        )
    message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))

    message_offset = len(file_to_use.stem) + 1
    data_frames = {
        filename.stem[message_offset:]: pd.read_csv(filename)
        for filename in message_csvs
    }
    first_timestamp = data_frames["vehicle_attitude_0"].iloc[0]["timestamp"]

    return {
        n: get_df_with_time(data_frames, m, first_timestamp)
        for n, m in NAME_TO_MESSAGES.items()
    }

def row_in_trajectory(row):
    """Check if a row belongs to a trajectory command."""
    return row["current.type"] == 8 and row["current.velocity_valid"]


def get_trajectory_bounds(dataframes, length, start_padding=0.0):
    """Get the trajectory time range if possible."""
    input_df = dataframes["commanded"]
    trajectory_times = []
    for i, time in enumerate(input_df["times"]):
        if row_in_trajectory(dataframes["commanded"].iloc[i]):
            trajectory_times.append(time)

    if len(trajectory_times) == 0:
        raise RuntimeError("Failed to find start time")

    start_time = min(trajectory_times) - start_padding
    return start_time, start_time + length


def lookup_time_index(df, t):
    """Get the index nearest to the time."""
    idx = bisect.bisect_right(df["times"], t)

    if idx == 0:
        return idx

    diff = df["times"][idx] - t
    other_diff = df["times"][idx - 1] - t

    if other_diff < diff:
        return idx - 1


def get_errors(dataframes, sample_times, bounds):
    """Get attitude errors for a specific run."""
    att = get_array_values(dataframes["attitude"], "q", (4,))
    att_sp = get_array_values(dataframes["attitude_sp"], "q_d", (4,))
    pos = get_field_values(dataframes["position"], ["x", "y", "z"], (3,))
    vel = get_field_values(dataframes["position"], ["vx", "vy", "vz"], (3,))
    pos_sp = get_field_values(dataframes["position_sp"], ["x", "y", "z"], (3,))
    vel_sp = get_field_values(dataframes["position_sp"], ["vx", "vy", "vz"], (3,))
    
#     att_err = []
    pos_arr = []
    pos_sp_arr = []
    vel_arr = []
    vel_sp_arr = []
    pos_err = []
    vel_err = []

    for t in sample_times:
        t_actual = t + bounds[0]

        p = pos[lookup_time_index(dataframes["position"], t_actual), :]
        p_sp = pos_sp[lookup_time_index(dataframes["position_sp"], t_actual), :]
        pos_arr.append(p)
        pos_sp_arr.append(p_sp)
        pos_err.append(abs(p - p_sp))

        v = vel[lookup_time_index(dataframes["position"], t_actual), :]
        v_sp = vel_sp[lookup_time_index(dataframes["position_sp"], t_actual), :]
        vel_arr.append(v)
        vel_sp_arr.append(v_sp)
        vel_err.append(abs(v - v_sp))

#         q = pyq.Quaternion(att[lookup_time_index(dataframes["attitude"], t_actual), :])
#         q_sp = pyq.Quaternion(
#             att_sp[lookup_time_index(dataframes["attitude_sp"], t_actual), :]
#         )
#         q_err = q.inverse * q_sp
#         att_err.append(abs(q_err.angle))

    return pos_arr, pos_sp_arr, pos_err, vel_arr, vel_sp_arr, vel_err

        
def plot_errors(ax, times, errors, label, **kwargs):
    mean = np.mean(errors, axis=1)
    stddev = np.std(errors, axis=1)
    handle = ax.plot(times, mean, label=label, **kwargs)[0]
    ax.fill_between(
        times, mean + stddev, mean - stddev, alpha=0.5, color=handle.get_color()
    )
    
def plot_sp(ax, times, errors, label, **kwargs):
    mean = np.mean(errors, axis=1)
    handle = ax.plot(times, mean, label=label, ls="--", **kwargs)[0]
    
def get_all_errors(start_time, length, samples, log_location, result_location):
    
    result_dir = pathlib.Path(result_location).resolve()

    files = get_files(log_location)

#     att_errors = np.zeros((N_samples, 3, len(files)))
    pos = np.zeros((N_samples, 3, len(files)))
    pos_sp = np.zeros((N_samples, 3, len(files)))
    pos_errors = np.zeros((N_samples, 3, len(files)))
    vel = np.zeros((N_samples, 3, len(files)))
    vel_sp = np.zeros((N_samples, 3, len(files)))
    vel_errors = np.zeros((N_samples, 3, len(files)))

    for i, filename in enumerate(files):
        dfs = get_dataframes(result_dir, filename)
        time_range = [start_time, start_time + length]
        errors = get_errors(dfs, samples, time_range)

#         att_errors[:, i] = np.array(errors[0])
        pos[:, :, i] = np.array(errors[0])
        pos_sp[:, :, i] = np.array(errors[1])
        pos_errors[:, :, i] = np.array(errors[2])
        vel[:, :, i] = np.array(errors[3])
        vel_sp[:, :, i] = np.array(errors[4])
        vel_errors[:, :, i] = np.array(errors[5])

    return pos, pos_sp, pos_errors, vel, vel_sp, vel_errors
    


# +
log_location = "../vision_test"
result_location = "../log_output"
start_time = 25
length = 15.0
N_samples = 60
samples = np.linspace(0.0, length, N_samples)

pos, pos_sp, pos_errors, vel, vel_sp, vel_errors = get_all_errors(start_time, length, samples, \
                                                    log_location, result_location)

fig, ax = plt.subplots(2, 2)

colors = sns.color_palette()
plot_errors(ax[0][0], samples, pos[:,0,:], label="x", color=colors[0])
plot_errors(ax[0][0], samples, pos[:,1,:], label="y", color=colors[1])
plot_errors(ax[0][0], samples, pos[:,2,:], label="z", color=colors[2])
plot_sp(ax[0][0], samples, pos_sp[:,0,:], label="x setpoint", color=colors[0])
plot_sp(ax[0][0], samples, pos_sp[:,1,:], label="y setpoint", color=colors[1])
plot_sp(ax[0][0], samples, pos_sp[:,2,:], label="z setpoint", color=colors[2])
ax[0][0].legend()
ax[0][0].set_title("Pepsi Bottle 2 m/s (Position)")
ax[0][0].set_xlabel("Time (seconds)")
ax[0][0].set_ylabel("Position (meters)")

plot_errors(ax[0][1], samples, pos_errors[:,0,:], label="x")
plot_errors(ax[0][1], samples, pos_errors[:,1,:], label="y")
plot_errors(ax[0][1], samples, pos_errors[:,2,:], label="z")
ax[0][1].legend()
ax[0][1].set_title("Pepsi Bottle 2 m/s Errors (Position)")
ax[0][1].set_xlabel("Time (seconds)")
ax[0][1].set_ylabel("Error (meters)")

plot_errors(ax[1][0], samples, vel[:,0,:], label="x", color=colors[0])
plot_errors(ax[1][0], samples, vel[:,1,:], label="y", color=colors[1])
plot_errors(ax[1][0], samples, vel[:,2,:], label="z", color=colors[2])
plot_sp(ax[1][0], samples, vel_sp[:,0,:], label="x setpoint", color=colors[0])
plot_sp(ax[1][0], samples, vel_sp[:,1,:], label="y setpoint", color=colors[1])
plot_sp(ax[1][0], samples, vel_sp[:,2,:], label="z setpoint", color=colors[2])
ax[1][0].legend()
ax[1][0].set_title("Pepsi Bottle 2 m/s (Velocity)")
ax[1][0].set_xlabel("Time (seconds)")
ax[1][0].set_ylabel("Velocity (meters/sec)")

plot_errors(ax[1][1], samples, vel_errors[:,0,:], label="x")
plot_errors(ax[1][1], samples, vel_errors[:,1,:], label="y")
plot_errors(ax[1][1], samples, vel_errors[:,2,:], label="z")
ax[1][1].legend()
ax[1][1].set_title("Pepsi Bottle 2 m/s Errors (Velocity)")
ax[1][1].set_xlabel("Time (seconds)")
ax[1][1].set_ylabel("Error (meters)")

fig.set_size_inches([25, 18])
plt.show()


# -
def load_bag(log_location, file_to_use, ds_period="10ms", useless_topic_fragments=None, bool_topics=None):
    
    if bool_topics is None:
        bool_topics = []
    if useless_topic_fragments is None:
        useless_topic_fragments = []

    run_bag = str(file_to_use)
#     result_dir = str(result_dir)
#     run_name = run_bag.split('/')[-1].split('.')[0]
#     run_base_dir = run_bag.split('/')[:-1]
#     data_dir = '/' + os.path.join(*run_base_dir, run_name)
#     run_bag = os.path.join(base, bag)
    run_name = run_bag.split('/')[-1].split('.')[0]
    run_base_dir = run_bag.split('/')[:-1]
    data_dir = '/' + os.path.join(*run_base_dir, run_name)
    print('checking %s' % log_location)
    if not os.path.exists(data_dir):
        print('Running csv creation on %s' % run_bag)
        b = bagreader(run_bag)
        csvfiles = []
        for t in b.topics:
            skip = False
            for p in useless_topic_fragments:
                if p in t:
                    skip = True
            if skip:
                continue
            data = b.message_by_topic(t)
            csvfiles.append(data)
    else:
        print('Already ran Bagpy')
        fns = os.listdir(data_dir)
        csvfiles = [os.path.join(data_dir, fn) for fn in fns]

    # Usually we shouldn't generate the extra csv files, but if we do have the csv files we don't want to load them because we run out of memory
    csvs_to_read = [fn for fn in csvfiles if not any([p in fn for p in useless_topic_fragments])]
    dfs = []
    for fn in csvs_to_read:
        print('Reading %s' % fn) 
        try:
            #dfs.append(pd.read_csv(fn))
            dfs.append(pd.read_csv(fn, quotechar='"'))
        except Exception as ex: 
            print(ex)
            continue
    for ix in range(len(dfs)):
        dfs[ix].Time = pd.to_datetime(dfs[ix].Time, unit='s')
        dfs[ix] = dfs[ix].set_index('Time')
        csv_fn = csvs_to_read[ix].split('/')[-1].split('.')[0]
        for c in dfs[ix].columns:
            dfs[ix] = dfs[ix].rename(columns={c:  csv_fn + '-' + c}) 
            if csv_fn in bool_topics:
                dfs[ix] = dfs[ix].astype(float)
    dfs_resampled = [d.select_dtypes(['number']).resample(ds_period).mean().interpolate(method='linear') for d in dfs] 
    df = pd.concat(dfs_resampled, join='outer', axis=1)
    # dfs is a list of Pandas dataframes, where each dataframe corresponds to a single topic
    # df is the combination of those dataframes after they have been resampled to the same time resolution
    return df



# +
def get_ulog_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    print(log_dir)
    return list(log_dir.glob("*.ulog"))
    

def get_rosbag_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return list(log_dir.glob("*.bag"))

def load_ulogs(result_location, log_location):
    result_dir = pathlib.Path(result_location).resolve()
    dfs = []
    files = get_files(log_location)
    print(files)
    for file in files:
        print(file)
        print(result_dir)
        df = get_dataframes(result_dir, file)
        dfs.append(df)
    return dfs
    

def load_rosbags(result_location, log_location):
    result_dir = pathlib.Path(result_location).resolve()
    dfs = []
    files = get_rosbag_files(log_location)
    for file in files:
        df = load_bag(result_dir, file, bool_topics=["grasp_state_machine_node-grasp_started"])
        dfs.append(df)
    return dfs


# -

log_location = "../vision_test"
log_result_location = "../vision_test"
bag_location = "../vision_test"
bag_result_location = "../vision_test"
rosbags = load_rosbags(bag_result_location, bag_location)
ulogs = load_ulogs(result_location, log_location)


# +


def get_bag_trajectory_bounds(bag):
    pass
    

def get_all_errors(bags, ulogs, length, samples):
    
#     att_errors = np.zeros((N_samples, 3, len(files)))
    pos = np.zeros((N_samples, 3, len(files)))
    pos_sp = np.zeros((N_samples, 3, len(files)))
    pos_errors = np.zeros((N_samples, 3, len(files)))
    vel = np.zeros((N_samples, 3, len(files)))
    vel_sp = np.zeros((N_samples, 3, len(files)))
    vel_errors = np.zeros((N_samples, 3, len(files)))

    for bag, ulog in zip(bags, ulogs):
        
        bag_bounds = get_bag_trajectory_bounds(bag)
        
        dfs = get_dataframes(result_dir, filename)
        time_range = [start_time, start_time + length]
        errors = get_errors(dfs, samples, time_range)

#         att_errors[:, i] = np.array(errors[0])
        pos[:, :, i] = np.array(errors[0])
        pos_sp[:, :, i] = np.array(errors[1])
        pos_errors[:, :, i] = np.array(errors[2])
        vel[:, :, i] = np.array(errors[3])
        vel_sp[:, :, i] = np.array(errors[4])
        vel_errors[:, :, i] = np.array(errors[5])

    return pos, pos_sp, pos_errors, vel, vel_sp, vel_errors

# fig, ax = plt.subplots(2, 2)
samples = np.linspace(0.0, 10, N_samples)
time_range = get_trajectory_bounds(ulogs[0], 10)
errors = get_errors(ulogs[1], samples, time_range)
# print(np.array(errors[0])[:,2])
plt.plot(samples, np.array(errors[0])[:,1], label="x", color=colors[0])
plt.plot(samples, np.array(errors[0])[:,0], label="y", color=colors[1])
plt.plot(samples, -np.array(errors[0])[:,2], label="z", color=colors[2])
plt.plot(samples, np.array(errors[1])[:,1], label="x", color=colors[0])
plt.plot(samples, np.array(errors[1])[:,0], label="y", color=colors[1])
plt.plot(samples, -np.array(errors[1])[:,2], label="z", color=colors[2])  
# rosbags[0]["grasp_state_machine_node-grasp_started-data"]
df = rosbags[0]
# cut = df[~df['grasp_state_machine_node-grasp_started-data'].isna()]
cut = df
start_time = 0 
end_time = cut.index[-1].timestamp() - cut.index[0].timestamp()
# print(len(cut))
cut["times"] = [cut.index[i].timestamp() - cut.index[0].timestamp() for i in range(len(cut))]
# print(cut["times"])
# cut.plot(x="times", y=["sparksdrone-world-pose.position.x",
#                        "sparksdrone-world-pose.position.y",
#                        "sparksdrone-world-pose.position.z",
#                        "grasp_state_machine_node-grasp_started-data",
#                        "cmd_gripper_sub-data",
#                       "sparkgrasptar-world-pose.position.x",
#                       "sparkgrasptar-world-pose.position.y",
#                       "sparkgrasptar-world-pose.position.z",
#                       "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x",
#                       "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y",
#                       "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z",
#                       "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.x",
#                       "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.y",
#                       "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.z"],
#         figsize=(20, 10))

cut["mocap_aligned_x"] = cut["gtsam_tracker_node_secondary-teaser_global-pose.position.x"] - cut["sparksdrone-world-pose.position.x"]
cut["mocap_aligned_y"] = cut["gtsam_tracker_node_secondary-teaser_global-pose.position.y"] - cut["sparksdrone-world-pose.position.y"]
cut["mocap_aligned_z"] = cut["gtsam_tracker_node_secondary-teaser_global-pose.position.z"] - cut["sparksdrone-world-pose.position.z"]

# cut.plot(x="times", y=["mocap_aligned_x",
#                        "mocap_aligned_y",
#                        "mocap_aligned_z",
#                       "gtsam_tracker_node-teaser_global-pose.position.x",
#                        "gtsam_tracker_node-teaser_global-pose.position.y",
#                        "gtsam_tracker_node-teaser_global-pose.position.z",
#                       "sparksdrone-world-pose.orientation.w",
#                       "sparksdrone-world-pose.orientation.x",
#                       "sparksdrone-world-pose.orientation.y",
#                       "sparksdrone-world-pose.orientation.z"],
#          figsize=(20,10))
                       
# cut.plot(x="times", y=["sparksdrone-world-pose.position.x",
#                        "sparksdrone-world-pose.position.y",
#                        "sparksdrone-world-pose.position.z",
#                        "grasp_state_machine_node-grasp_started-data",
#                        "gtsam_tracker_node-teaser_global-pose.position.x",
#                        "gtsam_tracker_node-teaser_global-pose.position.y",
#                        "gtsam_tracker_node-teaser_global-pose.position.z",
#                        "gtsam_tracker_node_secondary-teaser_global-pose.position.x",
#                        "gtsam_tracker_node_secondary-teaser_global-pose.position.y",
#                        "gtsam_tracker_node_secondary-teaser_global-pose.position.z"
                       
#                       ],
#         figsize=(20, 10))

cut.plot(x="times", y=["gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.x",
                       "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.y",
                       "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.z",
                      "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x",
                       "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y",
                       "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z"],
         figsize=(20,10))


log = ulogs[0]

# print(ulogs[0].keys())
# -

for col in cut.columns:
    print(col)

cut

df.plot(y=["cmd_gripper_sub-data", "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x", "sparksdrone-world-pose.position.x", 'grasp_state_machine_node-grasp_started-data'])

from load_bag import load_bag

dfs, df = load_bag("/home/subella/src/px4_ulog_analysis/test/", 
                   "1_2023-03-06-11-47-47.bag", 
                   "10ms", 
                   bool_topics=["grasp_state_machine_node/grasp_started"])

df.columns.tolist()



