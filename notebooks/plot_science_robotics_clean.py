# ---
# jupyter:
#   jupytext:
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
from natsort import natsorted
import os
import sys
import pandas as pd

from bagpy import bagreader
from natsort import natsorted
# -

sns.set()


def load_bag(log_location, file_to_use, ds_period="10ms", useless_topic_fragments=None, bool_topics=None):
    
    if bool_topics is None:
        bool_topics = []
    if useless_topic_fragments is None:
        useless_topic_fragments = []

    run_bag = str(file_to_use)
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
def get_rosbag_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return natsorted(list(log_dir.glob("*.bag")),key=str)

def load_rosbags(result_location, log_location):
    result_dir = pathlib.Path(result_location).resolve()
    dfs = []
    files = get_rosbag_files(log_location)
    for file in files:
        df = load_bag(result_dir, file, bool_topics=["grasp_state_machine_node-grasp_started"])
        dfs.append(df)
    return dfs


# -

def plot_std(ax, df, lower_df, upper_df, topics):
    for topic in topics:
        ax.fill_between(df["times"], 
                        lower_df[topic],
                        upper_df[topic], 
                        alpha=0.5)


# +
def get_trajectory_df(df, dt=0.01, start_padding=0, end_padding=0):
    cut = df[~df['grasp_state_machine_node-grasp_started-data'].isna()]
    cut["times"] = [dt * i for i in range(len(cut))]
    return cut

def get_trajectory_dfs(dfs):
    return [get_trajectory_df(df) for df in dfs]

def get_aggregate_df(dfs):
    aggregate_df = None
    for df in dfs:
        if aggregate_df is None:
            aggregate_df = df
        else:
            aggregate_df = pd.concat([aggregate_df, df])       
        
    mean_df = aggregate_df.groupby("times", as_index=False).mean()
    std_df = aggregate_df.groupby("times", as_index=False).std()
    return mean_df, std_df
        
def add_velocity(df, dt=0.01):
    df["sparksdrone-world-pose.velocity.x"] = (df["sparksdrone-world-pose.position.x"].diff().rolling(20).mean()) / dt
    df["sparksdrone-world-pose.velocity.y"] = (df["sparksdrone-world-pose.position.y"].diff().rolling(20).mean()) / dt
    df["sparksdrone-world-pose.velocity.z"] = (df["sparksdrone-world-pose.position.z"].diff().rolling(20).mean()) / dt
    df["sparkgrasptar-world-pose.velocity.x"] = (df["sparkgrasptar-world-pose.position.x"].diff().rolling(20).mean()) / dt
    df["sparkgrasptar-world-pose.velocity.y"] = (df["sparkgrasptar-world-pose.position.y"].diff().rolling(20).mean()) / dt
    df["sparkgrasptar-world-pose.velocity.z"] = (df["sparkgrasptar-world-pose.position.z"].diff().rolling(20).mean()) / dt

def add_velocities(dfs):
    for df in dfs:
        add_velocity(df)
    

# -

# # Mocap Turntable 0.5 m/s

ulog_location = "../mocap_turntable_05ms"
ulog_result_location = "../mocap_turntable_05ms"
bag_location = "../mocap_turntable_05ms"
bag_result_location = "../mocap_turntable_05ms"
rosbags = load_rosbags(bag_result_location, bag_location)

# +
traj_dfs = get_trajectory_dfs(rosbags)
evens_df = [traj_dfs[i] for i in range(0, len(traj_dfs), 2)]
odds_df = [traj_dfs[i] for i in range(1, len(traj_dfs), 2)]

add_velocities(evens_df)
evens_mean_df, evens_std_df = get_aggregate_df(evens_df)
evens_lower_df = evens_mean_df - evens_std_df
evens_upper_df = evens_mean_df + evens_std_df

add_velocities(odds_df)
odds_mean_df, odds_std_df = get_aggregate_df(odds_df)
odds_lower_df = odds_mean_df - odds_std_df
odds_upper_df = odds_mean_df + odds_std_df


# +
position_topics = ["sparksdrone-world-pose.position.x",
                   "sparksdrone-world-pose.position.y",
                   "sparksdrone-world-pose.position.z",
                   "sparkgrasptar-world-pose.position.x",
                   "sparkgrasptar-world-pose.position.y",
                   "sparkgrasptar-world-pose.position.z"]

velocity_topics = ["sparksdrone-world-pose.velocity.x",
                   "sparksdrone-world-pose.velocity.y",
                   "sparksdrone-world-pose.velocity.z",
                   "sparkgrasptar-world-pose.velocity.x",
                   "sparkgrasptar-world-pose.velocity.y",
                   "sparkgrasptar-world-pose.velocity.z",
                   "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.x",
                   "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.y",
                   "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.z"]

fig, ax = plt.subplots(1, 2, figsize=(15,5))
evens_mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], evens_mean_df, evens_lower_df, evens_upper_df, position_topics)
ax[0].legend(["Drone X", "Drone Y", "Drone Z", "Target X", "Target Y", "Target Z"])
# ax[0].axvline(x=evens_mean_df[, color='b', label='axvline - full height')

evens_mean_df.plot(x="times", y=velocity_topics, ax=ax[1])
plot_std(ax[1], evens_mean_df, evens_lower_df, evens_upper_df, velocity_topics)
ax[1].legend(["Drone Vx", "Drone Vy", "Drone Vz", "Target Vx", "Target Vy", "Target Vz"])


fig, ax = plt.subplots(1, 2, figsize=(15,5))
odds_mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], odds_mean_df, odds_lower_df, odds_upper_df, position_topics)
ax[0].legend(["Drone X", "Drone Y", "Drone Z", "Target X", "Target Y", "Target Z"])

evens_mean_df.plot(x="times", y=velocity_topics, ax=ax[1])
plot_std(ax[1], odds_mean_df, odds_lower_df, odds_upper_df, velocity_topics)
ax[1].legend(["Drone Vx", "Drone Vy", "Drone Vz", "Target Vx", "Target Vy", "Target Vz"])

# -

# # Mocap A1 Slow

ulog_location = "../mocap_a1_slow"
ulog_result_location = "../mocap_a1_slow"
bag_location = "../mocap_a1_slow"
bag_result_location = "../mocap_a1_slow"
rosbags = load_rosbags(bag_result_location, bag_location)

traj_dfs = get_trajectory_dfs(rosbags)
add_velocities(traj_dfs)
mean_df, std_df = get_aggregate_df(traj_dfs)
lower_df = mean_df - std_df
upper_df = mean_df + std_df

# +
position_topics = ["sparksdrone-world-pose.position.x",
                   "sparksdrone-world-pose.position.y",
                   "sparksdrone-world-pose.position.z",
                   "sparkgrasptar-world-pose.position.x",
                   "sparkgrasptar-world-pose.position.y",
                   "sparkgrasptar-world-pose.position.z"]

velocity_topics = ["sparksdrone-world-pose.velocity.x",
                   "sparksdrone-world-pose.velocity.y",
                   "sparksdrone-world-pose.velocity.z",
                   "sparkgrasptar-world-pose.velocity.x",
                   "sparkgrasptar-world-pose.velocity.y",
                   "sparkgrasptar-world-pose.velocity.z"]

fig, ax = plt.subplots(1, 2, figsize=(15,5))
mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], mean_df, lower_df, upper_df, position_topics)
ax[0].legend(["Drone X", "Drone Y", "Drone Z", "Target X", "Target Y", "Target Z"])

mean_df.plot(x="times", y=velocity_topics, ax=ax[1])
plot_std(ax[1], mean_df, lower_df, upper_df, velocity_topics)
ax[1].legend(["Drone Vx", "Drone Vy", "Drone Vz", "Target Vx", "Target Vy", "Target Vz"])
# -

# # Mocap A1 Fast

ulog_location = "../mocap_a1_fast"
ulog_result_location = "../mocap_a1_fast"
bag_location = "../mocap_a1_fast"
bag_result_location = "../mocap_a1_fast"
rosbags = load_rosbags(bag_result_location, bag_location)

traj_dfs = get_trajectory_dfs(rosbags)
add_velocities(traj_dfs)
mean_df, std_df = get_aggregate_df(traj_dfs)
lower_df = mean_df - std_df
upper_df = mean_df + std_df

# +
position_topics = ["sparksdrone-world-pose.position.x",
                   "sparksdrone-world-pose.position.y",
                   "sparksdrone-world-pose.position.z",
                   "sparkgrasptar-world-pose.position.x",
                   "sparkgrasptar-world-pose.position.y",
                   "sparkgrasptar-world-pose.position.z"]

velocity_topics = ["sparksdrone-world-pose.velocity.x",
                   "sparksdrone-world-pose.velocity.y",
                   "sparksdrone-world-pose.velocity.z",
                   "sparkgrasptar-world-pose.velocity.x",
                   "sparkgrasptar-world-pose.velocity.y",
                   "sparkgrasptar-world-pose.velocity.z"]

fig, ax = plt.subplots(1, 2, figsize=(15,5))
mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], mean_df, lower_df, upper_df, position_topics)
ax[0].legend(["Drone X", "Drone Y", "Drone Z", "Target X", "Target Y", "Target Z"])

mean_df.plot(x="times", y=velocity_topics, ax=ax[1])
plot_std(ax[1], mean_df, lower_df, upper_df, velocity_topics)
ax[1].legend(["Drone Vx", "Drone Vy", "Drone Vz", "Target Vx", "Target Vy", "Target Vz"])
# -

# # Vision Pepsi 2 m/s

ulog_location = "../vision_pepsi_2ms"
ulog_result_location = "../vision_pepsi_2ms"
bag_location = "../vision_pepsi_2ms"
bag_result_location = "../vision_pepsi_2ms"
rosbags = load_rosbags(bag_result_location, bag_location)

# +
board_to_target = [0.017, -0.183, 0.116]
fig, ax = plt.subplots(1, 1, figsize=(15,5))
for rosbag in rosbags:
    rosbag["times"] = [0.01 * i for i in range(len(rosbag))]
    rosbag["start"] = rosbag["grasp_state_machine_node-grasp_started-data"] - 4
    rosbag["aligned_mavros_y"] = -rosbag["mavros-odometry-out-pose.pose.position.y"]
    rosbag["target_x_gt"] = rosbag["sparkgrasptar-world-pose.position.x"] + board_to_target[0]
    rosbag["target_y_gt"] = rosbag["sparkgrasptar-world-pose.position.y"] + board_to_target[1]
    rosbag["target_z_gt"] = rosbag["sparkgrasptar-world-pose.position.z"] + board_to_target[2]
    rosbag["t265_to_mocap_x"] = rosbag["mavros-odometry-in-pose.pose.position.x"] + rosbag["sparksdrone-world-pose.position.x"].dropna()[0]
    rosbag["t265_to_mocap_y"] = rosbag["mavros-odometry-in-pose.pose.position.y"] + rosbag["sparksdrone-world-pose.position.y"].dropna()[1]
    rosbag["t265_to_mocap_z"] = rosbag["mavros-odometry-in-pose.pose.position.z"] + rosbag["sparksdrone-world-pose.position.z"].dropna()[2]
    rosbag.plot(x="times", y=["gtsam_tracker_node_secondary-teaser_global-pose.position.x",
                             "gtsam_tracker_node_secondary-teaser_global-pose.position.y",
                             "gtsam_tracker_node_secondary-teaser_global-pose.position.z"], ax=ax)
    
fig, ax = plt.subplots(1, 1, figsize=(15,5))

# +


mean_df, std_df = get_aggregate_df(rosbags)
lower_df = mean_df - std_df
upper_df = mean_df + std_df

position_topics = [
                "start",
#         "sparksdrone-world-pose.position.x",
#                    "sparksdrone-world-pose.position.y",
#                    "sparksdrone-world-pose.position.z",
#     "gtsam_tracker_node_secondary-teaser_global-pose.position.x",
                   "gtsam_tracker_node-teaser_global-pose.position.y",
    "mavros-odometry-in-pose.pose.position.y",
    "aligned_mavros_y",
#                    "gtsam_tracker_node_secondary-teaser_global-pose.position.z",
#                    "target_x_gt",
#                    "target_y_gt",
#                    "target_z_gt",
#                     "t265_to_mocap_y",
    "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y"
                  ]

fig, ax = plt.subplots(1, 1, figsize=(25,15))
rosbags[0].plot(x="times", y=position_topics, ax=ax)
plt.ylim([0.4, 0.7])
# plot_std(ax, mean_df, lower_df, upper_df, position_topics)
# ax[0].legend(["Target X", "Target Y", "Target Z"])



# +
position_topics = [
#     "sparksdrone-world-pose.position.x",
                   "sparksdrone-world-pose.position.y",
#                    "sparksdrone-world-pose.position.z",
#                    "mavros-odometry-in-pose.pose.position.x",
#                    "mavros-odometry-in-pose.pose.position.y",
#                    "mavros-odometry-in-pose.pose.position.z",
#                    "t265_to_mocap_x",
                   "t265_to_mocap_y",
#                    "t265_to_mocap_z",
                   
                  ]

fig, ax = plt.subplots(1, 2, figsize=(15,5))
mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], mean_df, lower_df, upper_df, position_topics)
ax[0].legend(["Mocap X", "Mocap Y", "Mocap Z", "T265 X", "T265 Y", "T265 Z"])


# +
position_topics = ["sparksdrone-world-pose.orientation.w",
                   "sparksdrone-world-pose.orientation.x",
                   "sparksdrone-world-pose.orientation.y",
                   
#                    "mavros-odometry-in-pose.pose.position.x",
#                    "mavros-odometry-in-pose.pose.position.y",
#                    "mavros-odometry-in-pose.pose.position.z",
                   "t265_to_mocap_x",
                   "t265_to_mocap_y",
                   "t265_to_mocap_z",
                   
                  ]

fig, ax = plt.subplots(1, 2, figsize=(15,5))
mean_df.plot(x="times", y=position_topics, ax=ax[0])
plot_std(ax[0], mean_df, lower_df, upper_df, position_topics)
ax[0].legend(["Mocap X", "Mocap Y", "Mocap Z", "T265 X", "T265 Y", "T265 Z"])

# +


mean_df, std_df = get_aggregate_df(rosbags)
lower_df = mean_df - std_df
upper_df = mean_df + std_df
position_topics = [
#                    "gtsam_tracker_node_secondary-teaser_global-pose.orientation.w",
                   "gtsam_tracker_node_secondary-teaser_global-pose.orientation.x",
                   "gtsam_tracker_node_secondary-teaser_global-pose.orientation.y",
    "gtsam_tracker_node_secondary-teaser_global-pose.orientation.z",
                  ]

fig, ax = plt.subplots(1, 1, figsize=(15,10))
mean_df.plot(x="times", y=position_topics, ax=ax)
plot_std(ax, mean_df, lower_df, upper_df, position_topics)
ax.legend(["Target w", "Target x", "Target y", "Target z"])


# -


