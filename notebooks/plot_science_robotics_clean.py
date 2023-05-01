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
import json

import evo.core.metrics as evo_metrics
import evo.core.lie_algebra as evo_lie
import evo.core.trajectory as evo
import evo.core.sync as evo_sync

from sklearn.preprocessing import normalize
from bagpy import bagreader
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
# -

sns.set()

# # Topic Shortcuts

# +
mocap_target_x = "sparkgrasptar-world-pose.position.x"
mocap_target_y = "sparkgrasptar-world-pose.position.y"
mocap_target_z = "sparkgrasptar-world-pose.position.z"

mocap_drone_x = "sparksdrone-world-pose.position.x"
mocap_drone_y = "sparksdrone-world-pose.position.y"
mocap_drone_z = "sparksdrone-world-pose.position.z"
mocap_drone_qw = "sparksdrone-world-pose.orientation.w"
mocap_drone_qx = "sparksdrone-world-pose.orientation.x"
mocap_drone_qy = "sparksdrone-world-pose.orientation.y"
mocap_drone_qz = "sparksdrone-world-pose.orientation.z"

mocap_aligned_x = "mocap_aligned_x"
mocap_aligned_y = "mocap_aligned_y"
mocap_aligned_z = "mocap_aligned_z"
mocap_aligned_qw = "mocap_aligned_qw"
mocap_aligned_qx = "mocap_aligned_qx"
mocap_aligned_qy = "mocap_aligned_qy"
mocap_aligned_qz = "mocap_aligned_qz"

mavros_in_x = "mavros-odometry-in-pose.pose.position.x"
mavros_in_y = "mavros-odometry-in-pose.pose.position.y"
mavros_in_z = "mavros-odometry-in-pose.pose.position.z"
mavros_in_qw = "mavros-odometry-in-pose.pose.orientation.w"
mavros_in_qx = "mavros-odometry-in-pose.pose.orientation.x"
mavros_in_qy = "mavros-odometry-in-pose.pose.orientation.y"
mavros_in_qz = "mavros-odometry-in-pose.pose.orientation.z"

mavros_out_x = "mavros-odometry-out-pose.pose.position.x"
mavros_out_y = "mavros-odometry-out-pose.pose.position.y"
mavros_out_z = "mavros-odometry-out-pose.pose.position.z"
mavros_out_qw = "mavros-odometry-out-pose.pose.orientation.w"
mavros_out_qx = "mavros-odometry-out-pose.pose.orientation.x"
mavros_out_qy = "mavros-odometry-out-pose.pose.orientation.y"
mavros_out_qz = "mavros-odometry-out-pose.pose.orientation.z"

mavros_in_aligned_x = "mavros_in_aligned_x"
mavros_in_aligned_y = "mavros_in_aligned_y"
mavros_in_aligned_z = "mavros_in_aligned_z"
mavros_in_aligned_qw = "mavros_in_aligned_qw"
mavros_in_aligned_qx = "mavros_in_aligned_qx"
mavros_in_aligned_qy = "mavros_in_aligned_qy"
mavros_in_aligned_qz = "mavros_in_aligned_qz"

mavros_gtsam_target_x = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x"
mavros_gtsam_target_y = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y"
mavros_gtsam_target_z = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z"

mocap_gtsam_target_x = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.x"
mocap_gtsam_target_y = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.y"
mocap_gtsam_target_z = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.z"


ulog_vo_x = "ulog_visual_odometry_x"
ulog_vo_y = "ulog_visual_odometry_y"
ulog_vo_z = "ulog_visual_odometry_z"
ulog_vo_qw = "ulog_visual_odometry_q[0]"
ulog_vo_qx = "ulog_visual_odometry_q[1]"
ulog_vo_qy = "ulog_visual_odometry_q[2]"
ulog_vo_qz = "ulog_visual_odometry_q[3]"

grasp_segment = "grasp_state_machine_node-grasp_started-data"
gripper_state = "cmd_gripper_sub-data"

mavros_drone_sp_x = "mavros-setpoint_raw-local-position.x"
mavros_drone_sp_y = "mavros-setpoint_raw-local-position.y"
mavros_drone_sp_z = "mavros-setpoint_raw-local-position.z"

# -

# # Ulog Helpers

# +
MESSAGES = [
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "position_setpoint_triplet",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "vehicle_rates_setpoint",
    "vehicle_visual_odometry"
]
MESSAGE_ARGS = ",".join(MESSAGES)

NAME_TO_MESSAGES = {
    "position": "vehicle_local_position_0",
    "position_sp": "vehicle_local_position_setpoint_0",
    "attitude": "vehicle_attitude_0",
    "attitude_sp": "vehicle_attitude_setpoint_0",
    "commanded": "position_setpoint_triplet_0",
    "attitude_output": "vehicle_rates_setpoint_0",
    "visual_odometry": "vehicle_visual_odometry_0"
}


# +
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


def lookup_time_index(df, t):
    """Get the index nearest to the time."""
    idx = bisect.bisect_right(df["times"], t)

    if idx == 0:
        return idx

    diff = df["times"][idx] - t
    other_diff = df["times"][idx - 1] - t

    if other_diff < diff:
        return idx - 1
    
def get_ulog_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return natsorted(list(log_dir.glob("*.ulg")))

def load_ulogs(result_location, log_location):
    result_dir = pathlib.Path(result_location).resolve()
    dfs = []
    files = get_ulog_files(log_location)
    for file in files:
        df = get_dataframes(result_dir, file)
        dfs.append(df)
    return dfs

def format_ulog_index(df):
    """Converts indices to time as float, starting at 0"""
    df.index = df["times"]
#     df.index -= df.index[0]
#     df["times"] = df.index
    
def format_ulogs_index(ulogs):
    for ulog in ulogs:
        format_ulog_index(ulog["position"])
        # First row of commanded is corrupted
        ulog["commanded"].drop([0])
        format_ulog_index(ulog["commanded"])
        format_ulog_index(ulog["visual_odometry"])
        
def reindex_ulogs(ulogs):
    for i in range(len(ulogs)):
        ulogs[i]["position"] = reindex_df(ulogs[i]["position"])
        ulogs[i]["commanded"] = reindex_df(ulogs[i]["commanded"], int_cols=["current.type"])
        ulogs[i]["visual_odometry"] = reindex_df(ulogs[i]["visual_odometry"])
        
def get_ulog_grasp_start_time(df):
    """Get the trajectory time range if possible."""
    return df[df["current.type"] == 8].index[0]

def center_ulog(ulog):
    start_time = get_ulog_grasp_start_time(ulog["commanded"])
    ulog["position"].index -= start_time
    ulog["commanded"].index -= start_time
    ulog["visual_odometry"].index -= start_time

def center_ulogs(ulogs):
    for ulog in ulogs:
        center_ulog(ulog)
    
def format_ulogs(ulogs):
    format_ulogs_index(ulogs)
    reindex_ulogs(ulogs)
    center_ulogs(ulogs)


# -

# # Rosbag Helpers

# +
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

def format_rosbag_index(rosbag, sample_rate=10):
    """Converts indexes to time as float, starting at 0"""
    rosbag.index = pd.to_timedelta(rosbag.index.strftime('%H:%M:%S.%f'))
    rosbag.index = rosbag.index.total_seconds()
#     print(rosbag.index)
#     rosbag.index -= rosbag.index[0]
#     rosbag["times"] = rosbag.index
    
def format_rosbags_index(rosbags, sample_rate=10):
    for rosbag in rosbags:
        rosbag = format_rosbag_index(rosbag, sample_rate)
        
def reindex_rosbags(rosbags, sample_rate=10):
    for i in range(len(rosbags)):
        rosbags[i] = reindex_df(rosbags[i], 
                                int_cols=[grasp_segment, gripper_state])
 
def get_rosbag_grasp_start_time(rosbag):
    return rosbag["grasp_state_machine_node-grasp_started-data"].dropna().index[0]

def center_rosbag(rosbag):
    rosbag.index -= get_rosbag_grasp_start_time(rosbag)

def center_rosbags(rosbags):
    for rosbag in rosbags:
        center_rosbag(rosbag)
        
def format_rosbags(rosbags, sample_rate=10):
    format_rosbags_index(rosbags, sample_rate)
    reindex_rosbags(rosbags, sample_rate)
    center_rosbags(rosbags)


# -

# # DataFrame Helpers

# +
def reindex_df(df, des_dt=0.01, int_cols=[]):
    float_cols = [x for x in df.columns if x not in int_cols]
    times = np.arange(round(df.index[0], 2), round(df.index[-1], 2), des_dt)
    # Convert to timedelta for time interpolation
    times = pd.to_timedelta(times, unit='s')
    df.index = pd.to_timedelta(df.index, unit='s')
    df = df.reindex(df.index.union(times))
    df.loc[:,float_cols] = df.loc[:,float_cols].interpolate(method='time')   
    df.loc[:,int_cols] = df.loc[:,int_cols].ffill()
    df = df.reindex(times)
    # Convert back to float for ease of use
    df.index = df.index.total_seconds()
    # Round so you can do floating point comparisons (prob should change this)
    df.index = np.round(df.index, 2)
    print(df.index)
    df["times"] = df.index
    return df

def merge_ulog_rosbag(ulog, rosbag):
    ulog["position"] = ulog["position"].add_prefix("ulog_position_")
    ulog["commanded"] = ulog["commanded"].add_prefix("ulog_commanded_")
    ulog["visual_odometry"] = ulog["visual_odometry"].add_prefix("ulog_visual_odometry_")
    
    rosbag.index = pd.to_timedelta(rosbag.index, unit='s')
    ulog["position"].index = pd.to_timedelta(ulog["position"].index, unit='s')
    ulog["commanded"].index = pd.to_timedelta(ulog["commanded"].index, unit='s')
    ulog["visual_odometry"].index = pd.to_timedelta(ulog["visual_odometry"].index, unit='s')
    
    df = pd.merge(rosbag, ulog["position"], left_index=True, right_index=True)
    df = pd.merge(df, ulog["commanded"], left_index=True, right_index=True)
    df = pd.merge(df, ulog["visual_odometry"], left_index=True, right_index=True)
    
    df.index = df.index.total_seconds()
    return df

def merge_ulogs_rosbags(ulogs, rosbags):
    dfs = []
    for (ulog, rosbag) in zip(ulogs, rosbags):
        dfs.append(merge_ulog_rosbag(ulog, rosbag))
    return dfs
    
def create_dfs(ulog_location, ulog_result_location, 
              bag_location, bag_result_location,
              sample_rate=0.1):
    
    rosbags = load_rosbags(bag_result_location, bag_location)
    format_rosbags(rosbags, sample_rate=10)
    
#     ulogs = load_ulogs(ulog_result_location, ulog_location)
#     format_ulogs(ulogs)
    
#     refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic="mavros-odometry-out-pose.pose.position.x")

#     dfs = merge_ulogs_rosbags(ulogs, rosbags)
    
#     return dfs
    return rosbags


# -

# # Trajectory Alignment

# +
def convert_traj_to_evo(positions, quaternions_wxyz, times):
    """Read a trajectory into the evo format."""
    return evo.PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quaternions_wxyz,
        timestamps=times,
    )

def get_aligned_trajectories(traj_curr, traj_ref, 
                             num_poses=None, min_ape_alignment=True, max_time_diff=0.3):
    """Get aligned trajectories."""
    traj_ref_synced, traj_curr_synced = evo_sync.associate_trajectories(
        traj_ref, traj_curr, max_diff=max_time_diff
    )

    if min_ape_alignment:
        ref_T_curr = traj_curr_synced.align(
            traj_ref_synced, correct_scale=False, n=num_poses
        )
    else:
        ref_H_curr = traj_curr_synced.align_origin(traj_ref_synced)
        ref_T_curr = (ref_H_curr[:3, :3], ref_H_curr[:3, 3], 1.0)
    return ref_T_curr, (traj_ref_synced, traj_curr_synced)

def refine_temporal_alignment_single(ulog, rosbag, rosbag_vo_x_topic):
    # Bag always lags behind ulog, so shift bag left until error is minimized
    step = 0.05
    shift = 0
    size = 200
    last_error = np.inf
    error = -np.inf
    while error < last_error:
        
        ulog_arr = -ulog["visual_odometry"]["y"].values
        bag_arr = rosbag[rosbag_vo_x_topic].values
        
        bag_center = np.where(rosbag.index == 0)[0][0]
        ulog_center = np.where(ulog["visual_odometry"].index == 0)[0][0]
        
        min_length = min(len(ulog_arr), len(bag_arr))
        if error != -np.inf:
            last_error = error
        error = np.sum(np.abs(ulog_arr[ulog_center - size: ulog_center + size] - bag_arr[bag_center - size: bag_center + size]))
        rosbag.index -= step
        rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
        
    # Add back step bc algo steps one too far    
    rosbag.index += step
    rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
    return rosbag

def refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic): 
    for i in range(len(rosbags)):
        rosbags[i] = refine_temporal_alignment_single(ulogs[i], rosbags[i], rosbag_vo_x_topic)



# -

# # Plotting Helpers

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

# # Trajectory Alignment

# +
ulog_location = "../vision_medkit_05ms"
ulog_result_location = "../log_output"
bag_location = "../vision_medkit_05ms"
bag_result_location = "../vision_medkit_05ms"

dfs = create_dfs(ulog_location, ulog_result_location, bag_location, bag_result_location)
# rosbags = load_rosbags(bag_result_location, bag_location)
# ulogs = load_ulogs(ulog_result_location, ulog_location)

# format_rosbags_index(rosbags)
# reindex_rosbags(rosbags)
# center_rosbags(rosbags)

# format_ulogs_index(ulogs)
# reindex_ulogs(ulogs)
# center_ulogs(ulogs)
# refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic="mavros-odometry-out-pose.pose.position.x")

# dfs = merge_ulogs_rosbags(ulogs, rosbags)


# +
df_sliced = dfs[2][-30:3]
# df_sliced.index = (df_sliced.index).astype(int)
print(df_sliced.index)
mavros_quat_topics = [mavros_in_qx, mavros_in_qy, mavros_in_qz, mavros_in_qw]
mocap_quat_topics = [mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]

mavros_quat = np.array(df_sliced.loc[:, mavros_quat_topics])
mavros_quat = mavros_quat / np.linalg.norm(mavros_quat, axis=1)[:,None]
mocap_quat = np.array(df_sliced.loc[:, mocap_quat_topics])
mocap_quat = mocap_quat / np.linalg.norm(mocap_quat, axis=1)[:,None]

df_sliced.loc[:, mavros_quat_topics] = mavros_quat
df_sliced.loc[:, mocap_quat_topics] = mocap_quat
df_sliced.index = ((30 + df_sliced.index) )
header = [mavros_in_x, mavros_in_y, mavros_in_z, mavros_in_qx, mavros_in_qy, mavros_in_qz, mavros_in_qw]
df_sliced.to_csv('poses_W_E.csv', columns = header, header=None)
header = [mocap_drone_x, mocap_drone_y, mocap_drone_z, mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]
df_sliced.to_csv('poses_B_H.csv', columns = header, header=None)

positions = []
rotations = []
for df in dfs:
    df_sliced = df[-30:3]
    mavros_quat_topics = [mavros_in_qx, mavros_in_qy, mavros_in_qz, mavros_in_qw]
    mocap_quat_topics = [mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]

    mavros_quat = np.array(df_sliced.loc[:, mavros_quat_topics])
    mavros_quat = mavros_quat / np.linalg.norm(mavros_quat, axis=1)[:,None]
    mocap_quat = np.array(df_sliced.loc[:, mocap_quat_topics])
    mocap_quat = mocap_quat / np.linalg.norm(mocap_quat, axis=1)[:,None]

    df_sliced.loc[:, mavros_quat_topics] = mavros_quat
    df_sliced.loc[:, mocap_quat_topics] = mocap_quat
    df_sliced.index = ((30 + df_sliced.index) )
    header = [mavros_in_x, mavros_in_y, mavros_in_z, mavros_in_qx, mavros_in_qy, mavros_in_qz, mavros_in_qw]
    df_sliced.to_csv('poses_W_E.csv', columns = header, header=None)
    header = [mocap_drone_x, mocap_drone_y, mocap_drone_z, mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]
    df_sliced.to_csv('poses_B_H.csv', columns = header, header=None)
    
    # !rosrun hand_eye_calibration compute_complete_handeye_calibration.sh poses_B_H.csv poses_W_E.csv 

    with open("calibration_optimized.json", 'r') as f:
        data = json.load(f)
        # !rm calibration_optimized.json
        p = [ float(data['translation'][name]) for name in 'xyz' ] 
        q = np.array([ float(data['rotation'][name]) for name in 'ijkw' ])
        positions.append(p)
        rotations.append(q)

# +
# print(np.mean(positions, axis=0))
# print(np.std(positions, axis=0))
# print(np.mean(rotations, axis=0))
# print(np.std(rotations, axis=0))

rotations_flip = [r if r[0] > 0 else -r for r in rotations ]
for r in rotations_flip:
    print(r)
    
print("mean",  np.mean(rotations_flip, axis=0))
print("std", np.std(rotations_flip, axis=0))
# -

in_rotations = rotations_flip.copy()

# +

df_sliced = dfs[7][-30:3]
mavros_quat_topics = [mavros_out_qx, mavros_out_qy, mavros_out_qz, mavros_out_qw]
mocap_quat_topics = [mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]

mavros_quat = np.array(df_sliced.loc[:, mavros_quat_topics])
mavros_quat = mavros_quat / np.linalg.norm(mavros_quat, axis=1)[:,None]
mocap_quat = np.array(df_sliced.loc[:, mocap_quat_topics])
mocap_quat = mocap_quat / np.linalg.norm(mocap_quat, axis=1)[:,None]

# print("mavros_quat", np.array(df_sliced.loc[:, mavros_quat]))
# print("norm", np.linalg.norm(df_sliced.loc[:, mavros_quat], axis=1)[:,None])
# print("mavros quat normed", np.array(df_sliced.loc[:, mavros_quat]) / np.linalg.norm(df_sliced.loc[:, mavros_quat], axis=1)[:,None])
# print(np.linalg.norm(np.array(df_sliced.loc[:,[mavros_out_qx, mavros_out_qy, mavros_out_qz, mavros_out_qw]]), axis=1))
df_sliced.loc[:, mavros_quat_topics] = mavros_quat
df_sliced.loc[:, mocap_quat_topics] = mocap_quat
df_sliced.index = ((30 + df_sliced.index) )
header = [mavros_out_x, mavros_out_y, mavros_out_z, mavros_out_qx, mavros_out_qy, mavros_out_qz, mavros_out_qw]
df_sliced.to_csv('poses_W_E.csv', columns = header, header=None)
header = [mocap_drone_x, mocap_drone_y, mocap_drone_z, mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]
df_sliced.to_csv('poses_B_H.csv', columns = header, header=None)

# +
# for df in dfs:
fig, ax = plt.subplots(1, 1, figsize=(15,5))
#     ax.plot(ulog["commanded"].index, ulog["commanded"]["current.type"], label="Ulog x")
# #     ax.plot(ulog["visual_odometry"].index, -ulog["visual_odometry"]["x"], label="Ulog x")
# ax.plot(df.index, -df["ulog_visual_odometry_y"], label="Ulog y")
# ax.plot(df.index, -df["ulog_visual_odometry_z"], label="Ulog z")
# ax.plot(df.index, -df["ulog_visual_odometry_x"], label="Ulog pos x")
# #     ax.plot(ulog["position"].index, -ulog["position"]["y"], label="Ulog pos y")
# #     ax.plot(ulog["position"].index, -ulog["position"]["z"], label="-Ulog pos z")
# ax.plot(df.index, df["mavros-odometry-out-pose.pose.position.x"], label="Bag x")
# ax.plot(df.index, df["mavros-odometry-out-pose.pose.position.y"], label="Bag y")
# ax.plot(df.index, df["mavros-odometry-out-pose.pose.position.z"], label="Bag z")
# ax.plot(df.index, df["grasp_state_machine_node-grasp_started-data"], label="Bag z")

ax.plot(df.index, df[mavros_in_qx], label="mavros x")
ax.plot(df.index, df[mavros_in_qy], label="mavros y")
ax.plot(df.index, df[mavros_in_qz], label="mavros z")
ax.plot(df.index, df[mavros_in_qw], label="mavros qw")
quat = np.array(df.loc[:, mavros_quat_topics])
r = R.from_quat(quat[0]).as_euler("xyz", degrees=True)
print(r)
# ax.plot(df.index, df[mocap_drone_x], label="mocap x")
# ax.plot(df.index, df[mocap_drone_y], label="mocap y")
# ax.plot(df.index, df[mocap_drone_z], label="mocap z")

# ax.plot(df.index, df["ulog_commanded_current.type"], label="Bag z")
#     ax.set_xlim(right=25)
ax.legend()


# +
def align_mavros_to_mocap(df, start_time=0, num_poses=None):
    # Start alignment at grasp start before grasp start
    df_sliced = df.loc[start_time:]
#     df_sliced = df
    times = np.array(df_sliced.index)
    mavros_positions = np.array([df_sliced[mavros_in_x], 
                                 df_sliced[mavros_in_y],
                                 df_sliced[mavros_in_z]]).T
    mavros_quaternions_wxyz = np.array([df_sliced[mavros_in_qw],
                                        df_sliced[mavros_in_qx], 
                                        df_sliced[mavros_in_qy], 
                                        df_sliced[mavros_in_qz]]).T

#     mavros_positions = np.array([df_sliced[ulog_vo_x], 
#                                  df_sliced[ulog_vo_y],
#                                  df_sliced[ulog_vo_z]]).T
#     mavros_quaternions_wxyz = np.array([df_sliced[ulog_vo_qw],
#                                         df_sliced[ulog_vo_qx], 
#                                         df_sliced[ulog_vo_qy], 
#                                         df_sliced[ulog_vo_qz]]).T

    mocap_positions = np.array([df_sliced[mocap_drone_x], 
                                df_sliced[mocap_drone_y],
                                df_sliced[mocap_drone_z]]).T
    mocap_quaternions_wxyz = np.array([df_sliced[mocap_drone_qw],
                                       df_sliced[mocap_drone_x], 
                                       df_sliced[mocap_drone_y], 
                                       df_sliced[mocap_drone_z]]).T
    
#     mavros_quaternions_wxyz = mocap_quaternions_wxyz
#     print("mavros", mavros_quaternions_wxyz)
#     print("mocap", mocap_quaternions_wxyz)
    
    mavros_evo = convert_traj_to_evo(mavros_positions, mavros_quaternions_wxyz, times)
    mocap_evo = convert_traj_to_evo(mocap_positions, mocap_quaternions_wxyz, times)

    T, (mocap_aligned, mavros_aligned) = get_aligned_trajectories(mavros_evo, mocap_evo, num_poses=num_poses)

    return T, mocap_aligned, mavros_aligned

def make_tf(pos, rot):
#     r = R.from_quat(rot).as_matrix()
    T = np.zeros((4,4))
    T[:3,:3] = rot
    T[:3, 3] = pos.T
    T[3, :] = [0,0,0,1]
    return T


def compute_marker_wrt_px4(df, mavros_wrt_mocap):
    for index, row in df.iterrows():
        px4_wrt_mavros_trans = np.array([row[mavros_in_x],
                                         row[mavros_in_y],
                                         row[mavros_in_z]])
        px4_wrt_mavros_rot = np.array([row[mavros_in_qx],
                                         row[mavros_in_qy],
                                         row[mavros_in_qz],
                                         row[mavros_in_qw]])
        px4_wrt_mavros_rot = R.from_quat(px4_wrt_mavros_rot).as_matrix()

        marker_wrt_mocap_trans = np.array([row[mocap_drone_x],
                                         row[mocap_drone_y],
                                         row[mocap_drone_z]])
        marker_wrt_mocap_rot = np.array([row[mocap_drone_qx],
                                         row[mocap_drone_qy],
                                         row[mocap_drone_qz],
                                         row[mocap_drone_qw]])
        marker_wrt_mocap_rot = R.from_quat(marker_wrt_mocap_rot).as_matrix()


        px4_wrt_mavros = make_tf(px4_wrt_mavros_trans, px4_wrt_mavros_rot)
        marker_wrt_mocap = make_tf(marker_wrt_mocap_trans, marker_wrt_mocap_rot)

#         print("mavros_wrt_mocap", mavros_wrt_mocap)
#         print("px4_wrt_mavros", px4_wrt_mavros)
#         print("marker_wrt_mocap", marker_wrt_mocap)
        
        px4_wrt_mocap = mavros_wrt_mocap.dot(px4_wrt_mavros)
#         print("px4_wrt_mocap", px4_wrt_mocap)
        marker_wrt_px4 = np.linalg.inv(marker_wrt_mocap).dot(px4_wrt_mocap)
#         print("marker_wrt_px4", marker_wrt_px4)
        return marker_wrt_px4


#         target_est_transform_df.loc[index, 't265_wrt_mocap_x'] = t265_wrt_mocap[0,3]
#         target_est_transform_df.loc[index, 't265_wrt_mocap_y'] = t265_wrt_mocap[1,3]
#         target_est_transform_df.loc[index, 't265_wrt_mocap_z'] = t265_wrt_mocap[2,3]

#     #     target_est_transform_df.loc[index, 'target_est_x'] = target_wrt_mocap[0,3]
#     #     target_est_transform_df.loc[index, 'target_est_y'] = target_wrt_mocap[1,3]
#     #     target_est_transform_df.loc[index, 'target_est_z'] = target_wrt_mocap[2,3]

#         target_est_transform_df.loc[index, 'target_est_x'] = target_wrt_mocap[0,3]
#         target_est_transform_df.loc[index, 'target_est_y'] = target_wrt_mocap[1,3]
#         target_est_transform_df.loc[index, 'target_est_z'] = target_wrt_mocap[2,3]

#     #     print("t265_wrt_mocap", t265_wrt_mocap)
#     #     print("target_wrt_mocap", target_wrt_mocap)
#     #     print("Est", target_wrt_mocap[:3, 3].T)
#     #     print("GT", target_position_mocap)
#         print("tar_wrt_t265", target_wrt_t265)
#         print("drone_wrt_t265", drone_wrt_t265)
#         print("drone_wrt_mocap", drone_wrt_mocap)
#         print("error", target_wrt_mocap[:3, 3].T - target_position_mocap)

def get_marker_wrt_px4(dfs):
    num_samples = 1
    num_poses=600
    marker_wrt_px4_tfs = np.zeros((len(dfs) * num_samples,4,4))
    xs = []
    ys = []
    zs = []
    for i, df in enumerate(dfs):

        for j in range(num_samples):
            df_sliced = df[0:num_poses/100]
#             print(df_sliced.index)
            T, _, mavros_aligned = align_mavros_to_mocap(df_sliced, start_time=0)
            mavros_wrt_mocap = make_tf(T[1], T[0])
            marker_wrt_px4 = compute_marker_wrt_px4(df_sliced, mavros_wrt_mocap)
            xs.append(mavros_wrt_mocap[0,3])
            ys.append(mavros_wrt_mocap[1,3])
            zs.append(mavros_wrt_mocap[2,3])

            marker_wrt_px4_tfs[i*num_samples + j,:,:] = marker_wrt_px4

            aligned_df = pd.DataFrame()
#             df_sliced = df.loc[-j:]
            aligned_df.index = df_sliced.index
            aligned_df[mavros_in_aligned_x] = mavros_aligned.positions_xyz[:,0]
            aligned_df[mavros_in_aligned_y] = mavros_aligned.positions_xyz[:,1]
            aligned_df[mavros_in_aligned_z] = mavros_aligned.positions_xyz[:,2]
            aligned_df[mavros_in_aligned_qw] = mavros_aligned.orientations_quat_wxyz[:,0]
            aligned_df[mavros_in_aligned_qx] = mavros_aligned.orientations_quat_wxyz[:,1]
            aligned_df[mavros_in_aligned_qy] = mavros_aligned.orientations_quat_wxyz[:,2]
            aligned_df[mavros_in_aligned_qz] = mavros_aligned.orientations_quat_wxyz[:,3]
            
            print(aligned_df[mavros_in_aligned_qx][0:])
            dff = df.join(aligned_df)
            print(dff[mavros_in_aligned_qx][0:])
            fig, ax = plt.subplots(2, 2, figsize=(15,5))


            ax[0][0].plot(dff.index, dff[mocap_drone_x], label="mocap x")
            ax[0][0].plot(dff.index, dff[mocap_drone_y], label="mocap y")
            ax[0][0].plot(dff.index, dff[mocap_drone_z], label="mocap z")

            ax[0][0].plot(dff.index, dff[mavros_in_aligned_x], label="mavros x")
            ax[0][0].plot(dff.index, dff[mavros_in_aligned_y], label="mavros y")
            ax[0][0].plot(dff.index, dff[mavros_in_aligned_z], label="mavros z")
            
            ax[0][1].plot(dff[mocap_drone_x], dff[mocap_drone_y], label="mocap xy")
            ax[0][1].plot(dff[mavros_in_aligned_x], dff[mavros_in_aligned_y], label="mavros xy")
            
            start_id = max(dff[mavros_in_aligned_qx].dropna().index[0], dff[mocap_drone_qx].dropna().index[0])
            end_id = min(dff[mavros_in_aligned_qx].dropna().index[-1], dff[mocap_drone_qx].dropna().index[-1])



            df_sliced = dff[start_id:end_id]
            mavros_quat_topics = [mavros_in_aligned_qx, mavros_in_aligned_qy, mavros_in_aligned_qz, mavros_in_aligned_qw]
            mocap_quat_topics = [mocap_drone_qx, mocap_drone_qy, mocap_drone_qz, mocap_drone_qw]

            mavros_quat = np.array(df_sliced.loc[:, mavros_quat_topics])
            mavros_quat = mavros_quat / np.linalg.norm(mavros_quat, axis=1)[:,None]
            mocap_quat = np.array(df_sliced.loc[:, mocap_quat_topics])
            mocap_quat = mocap_quat / np.linalg.norm(mocap_quat, axis=1)[:,None]

        #     print(mavros_quat)
            mavros_rpy = R.from_quat(mavros_quat).as_euler("xyz", degrees=True)
        #     print(mavros_rpy.shape)
        #     print(medkit_df.index)
            mocap_rpy = R.from_quat(mocap_quat).as_euler("xyz", degrees=True)
            
            ax[1][1].plot(dff[start_id:end_id].index, mavros_rpy[:,2], label="mavros yaw")
            ax[1][1].plot(dff[start_id:end_id].index, mocap_rpy[:,2], label="mocap yaw")
            ax[1][1].legend()
            
            ax[1][0].plot(dff[start_id:end_id].index, mocap_quat[:,0], label="mocap qx")
            ax[1][0].plot(dff[start_id:end_id].index, mocap_quat[:,1], label="mocap qy")
            ax[1][0].plot(dff[start_id:end_id].index, mocap_quat[:,2], label="mocap qz")
            ax[1][0].plot(dff[start_id:end_id].index, mocap_quat[:,3], label="mocap qw")
    
            
            ax[0][1].set_xlim(-0.5, 0.5)
            ax[1][1].set_xlim(-0.5, 6)
            ax[1][1].set_ylim(-10,10)
            ax[1][0].set_ylim(0, 0.25)
#             ax.plot(dff.index, dff[mavros_in_aligned_z], label="mavros z")

    marker_wrt_px4_mean = np.mean(marker_wrt_px4_tfs, axis=(0))
    marker_wrt_px4_std = np.std(marker_wrt_px4_tfs, axis=(0))
    print("Computed marker_wrt_px4 mean", marker_wrt_px4_mean)
    print("Computed marker_wrt_px4 std", marker_wrt_px4_std)
    print("rots", marker_wrt_px4_tfs)
    print("eulers", R.from_matrix(marker_wrt_px4_tfs[:,:3,:3]).as_euler("xyz", degrees=True))
    print("rot", R.from_matrix(marker_wrt_px4_mean[:3,:3]).as_euler("xyz", degrees=True))
    print("rot", np.std(R.from_matrix(marker_wrt_px4_tfs[:,:3,:3]).as_euler("xyz", degrees=True), axis=0))
    return marker_wrt_px4_mean, xs, ys, zs
            
# -

marker_wrt_px4, xs, ys, zs = get_marker_wrt_px4(dfs)

counts, bins = np.histogram(xs)
plt.plot(xs)

print(np.mean(tfs, axis=(0)))
print(np.std(tfs, axis=(0)))
print(marker_wrt_px4)

# +
for df in dfs:
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    T , mocap_aligned, mavros_aligned = align_mavros_to_mocap(df)
    aligned_df = pd.DataFrame()
    df_sliced = df.loc[0:]
    aligned_df.index = df_sliced.index
    aligned_df[mavros_in_aligned_x] = mavros_aligned.positions_xyz[:,0]
    aligned_df[mavros_in_aligned_y] = mavros_aligned.positions_xyz[:,1]
    aligned_df[mavros_in_aligned_z] = mavros_aligned.positions_xyz[:,2]
    aligned_df[mavros_in_aligned_qw] = mavros_aligned.orientations_quat_wxyz[:,0]
    aligned_df[mavros_in_aligned_qx] = mavros_aligned.orientations_quat_wxyz[:,1]
    aligned_df[mavros_in_aligned_qy] = mavros_aligned.orientations_quat_wxyz[:,2]
    aligned_df[mavros_in_aligned_qz] = mavros_aligned.orientations_quat_wxyz[:,3]
    df = df.join(aligned_df)

#     ax.plot(df.index, df[mocap_drone_x], label="mocap x")
    ax.plot(df.index, df[mocap_drone_y], label="mocap y")
#     ax.plot(df.index, df[mocap_drone_z], label="mocap z")

#     ax.plot(df.index, df[mocap_aligned_x], label="mocap x")
#     ax.plot(df.index, df[mocap_aligned_y], label="mocap y")
#     ax.plot(df.index, df[mocap_aligned_z], label="mocap z")

#     ax.plot(df.index, df[mavros_in_x], label="mavros x")
#     ax.plot(df.index, df[mavros_in_y], label="mavros y")
#     ax.plot(df.index, df[mavros_in_z], label="mavros z")

#     ax.plot(df.index, df[mavros_in_aligned_x], label="mavros x")
    ax.plot(df.index, df[mavros_in_aligned_y], label="mavros y")
#     ax.plot(df.index, df[mavros_in_aligned_z], label="mavros z")
    
#     ax.plot(df.index, df[mavros_drone_sp_x], label="mavros sp x")
    ax.plot(df.index, df[mavros_drone_sp_y], label="mavros sp y")
#     ax.plot(df.index, df[mavros_drone_sp_z], label="mavros sp z")
    
    ax.plot(df.index, df[gripper_state])


    # ax.plot(dfs[0].index, dfs[0][mocap_drone_qw], label="mocap qw")
    # ax.plot(dfs[0].index, dfs[0][mocap_drone_qx], label="mocap qx")
    # ax.plot(dfs[0].index, dfs[0][mocap_drone_qy], label="mocap qy")
    # ax.plot(dfs[0].index, dfs[0][mocap_drone_qz], label="mocap qz")

    # ax.plot(dfs[0].index, dfs[0][mavros_in_qw], label="mavros qw")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_qx], label="mavros qx")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_qy], label="mavros qy")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_qz], label="mavros qz")

    # ax.plot(dfs[0].index, dfs[0][mavros_in_aligned_qw], label="mavros aligned qw")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_aligned_qx], label="mavros aligned qx")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_aligned_qy], label="mavros aligned qy")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_aligned_qz], label="mavros aligned qz")

    ax.legend()
    ax.set_xlim(left=0)

#     print(dfs[0])
# -

dfs[0][0]

# +
from scipy.spatial.transform import Rotation as R



def get_mocap_wrt_t265(df):
    # Returns transform of mocap in t26 frame by running evo over short period
    pass
    
    
for i, df in enumerate(dfs):
    target_est_transform_df = pd.DataFrame()
    df_sliced = df[0:]
    T, _, mavros_aligned = align_mavros_to_mocap(df, start_time=0, num_poses=20)
    mavros_wrt_mocap = make_tf(T[1], T[0])
    
#     aligned_df = pd.DataFrame()
# #     df_sliced = df.loc[0:]
#     aligned_df.index = df[0:].index
#     aligned_df[mavros_in_aligned_x] = mavros_aligned.positions_xyz[:,0]
#     aligned_df[mavros_in_aligned_y] = mavros_aligned.positions_xyz[:,1]
#     aligned_df[mavros_in_aligned_z] = mavros_aligned.positions_xyz[:,2]
#     aligned_df[mavros_in_aligned_qw] = mavros_aligned.orientations_quat_wxyz[:,0]
#     aligned_df[mavros_in_aligned_qx] = mavros_aligned.orientations_quat_wxyz[:,1]
#     aligned_df[mavros_in_aligned_qy] = mavros_aligned.orientations_quat_wxyz[:,2]
#     aligned_df[mavros_in_aligned_qz] = mavros_aligned.orientations_quat_wxyz[:,3]
#     dff = df.join(aligned_df)
#     fig, ax = plt.subplots(1, 1, figsize=(15,5))

#     ax.plot(dff.index, dff[mavros_in_aligned_x] - dff[mocap_drone_x], label="mocap x")
#     ax.plot(dff.index, dff[mavros_in_aligned_y] - dff[mocap_drone_y], label="mocap y")
#     ax.plot(dff.index, dff[mavros_in_aligned_z] - dff[mocap_drone_z], label="mocap z")
#     ax.set_ylim([-.02,.02])
#     ax.plot(dff.index, dff[mocap_drone_x], label="mocap x")
#     ax.plot(dff.index, dff[mocap_drone_y], label="mocap y")
#     ax.plot(dff.index, dff[mocap_drone_z], label="mocap z")

#     ax.plot(dff.index, dff[mavros_in_aligned_x], label="mavros x")
#     ax.plot(dff.index, dff[mavros_in_aligned_y], label="mavros y")
#     ax.plot(dff.index, dff[mavros_in_aligned_z], label="mavros z")
    
    
    for index, row in df.iterrows():
#         break
        
        
    #     print(row)
        target_position_t265 = np.array([row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x"],
                                         row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y"],
                                         row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z"]])

        if np.isnan(target_position_t265).any():
            continue

        target_rotation_t265 = np.array([row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.x"],
                                         row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.y"],
                                         row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.z"],
                                         row["gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.w"]])
        target_rotation_t265 = R.from_quat(target_rotation_t265).as_matrix()

        drone_position_t265 = np.array([row["mavros-odometry-in-pose.pose.position.x"],
                                         row["mavros-odometry-in-pose.pose.position.y"],
                                         row["mavros-odometry-in-pose.pose.position.z"]])
        drone_rotation_t265 = np.array([row["mavros-odometry-in-pose.pose.orientation.x"],
                                         row["mavros-odometry-in-pose.pose.orientation.y"],
                                         row["mavros-odometry-in-pose.pose.orientation.z"],
                                         row["mavros-odometry-in-pose.pose.orientation.w"]])
        drone_rotation_t265 = R.from_quat(drone_rotation_t265).as_matrix()

        drone_position_mocap = np.array([row["sparksdrone-world-pose.position.x"],
                                         row["sparksdrone-world-pose.position.y"],
                                         row["sparksdrone-world-pose.position.z"]])
        drone_rotation_mocap = np.array([row["sparksdrone-world-pose.orientation.x"],
                                         row["sparksdrone-world-pose.orientation.y"],
                                         row["sparksdrone-world-pose.orientation.z"],
                                         row["sparksdrone-world-pose.orientation.w"]])
        drone_rotation_mocap = R.from_quat(drone_rotation_mocap).as_matrix()

        target_position_mocap = np.array([row["sparkgrasptar-world-pose.position.x"],
                                         row["sparkgrasptar-world-pose.position.y"],
                                         row["sparkgrasptar-world-pose.position.z"]])

        
        T, _, mavros_aligned = align_mavros_to_mocap(df[index-4:index+4], start_time=index-4)
#         mavros_wrt_mocap = make_tf(T[1], T[0])



        target_wrt_mavros = make_tf(target_position_t265, target_rotation_t265)
        px4_wrt_mavros = make_tf(drone_position_t265, drone_rotation_t265)
        marker_wrt_mocap = make_tf(drone_position_mocap, drone_rotation_mocap)

#         q = [
#              0.3221715516423071,
#              0.013395667410090696,
#              -0.94657416484160961,
#             0.0048371335970901425,
#              ]
#         e = [1.01, -0.042, -2.965 ]
# #         r = R.from_quat(q).as_euler("xyz", degrees=True)
#         r = R.from_euler("xyz", e, degrees=False).as_quat()
#         print(r)
#         t = np.array([-0.019, -0.036, -0.003])
#         marker_wrt_px4 = make_tf(t, r)
        target_wrt_mocap = marker_wrt_mocap.dot(np.linalg.inv(marker_wrt_px4)).dot(np.linalg.inv(px4_wrt_mavros)).dot(target_wrt_mavros)

#         target_wrt_mocap = mavros_wrt_mocap.dot(target_wrt_mavros) 



        target_est_transform_df.loc[index, 'target_est_x'] = target_wrt_mocap[0,3]
        target_est_transform_df.loc[index, 'target_est_y'] = target_wrt_mocap[1,3]
        target_est_transform_df.loc[index, 'target_est_z'] = target_wrt_mocap[2,3]


    #     print("t265_wrt_mocap", t265_wrt_mocap)
#         print("marker_wrt_mocap", marker_wrt_mocap)
        print("marker_wrt_px4", marker_wrt_px4)
#         print("px4_wrt_mavros", px4_wrt_mavros)
        print("target_wrt_mavros", target_wrt_mavros)
        print("mavros_wrt_mocap", mavros_wrt_mocap)
        print("target_wrt_mocap", target_wrt_mocap)
        print("target actual mocap", target_position_mocap)
    #     print("Est", target_wrt_mocap[:3, 3].T)
    #     print("GT", target_position_mocap)
    #     print("tar_wrt_t265", target_wrt_t265)
    #     print("drone_wrt_t265", drone_wrt_t265)
    #     print("drone_wrt_mocap", drone_wrt_mocap)

        print("error", target_wrt_mocap[:3, 3].T - target_position_mocap)

    dfs[i] = dfs[i].join(target_est_transform_df)
# dfs[0] = pd.merge(dfs[0], target_est_transform_df, left_index=True, right_index=True)
    
# -

# target_est_transform_df
dfs[0].index

# +
for df in dfs:
    fig, ax = plt.subplots(1, 1, figsize=(15,5))

    # ax.plot(dfs[0].index, dfs[0][mavros_gtsam_target_x], label="t265 est x")
#     ax.plot(df.index, df[mavros_gtsam_target_y], label="t265 est y")
    # ax.plot(dfs[0].index, dfs[0][mavros_gtsam_target_z], label="t265 est z")
    
    ax.plot(df.index, df[mocap_gtsam_target_x], label="mocap gtsam x")
    ax.plot(df.index, df[mocap_gtsam_target_y], label="mocap gtsam y")
    ax.plot(df.index, df[mocap_gtsam_target_z], label="mocap gtsam z")

    ax.plot(df.index, df[mocap_target_x], label="mocap x")
    ax.plot(df.index, df[mocap_target_y], label="mocap y")
    ax.plot(df.index, df[mocap_target_z], label="mocap z")

#     ax.plot(df.index, df["target_est_x"], label="mocap est x")
#     ax.plot(df.index, df["target_est_y"], label="mocap est y")
#     ax.plot(df.index, df["target_est_z"], label="mocap est z")

#     ax.plot(df.index, np.abs(df["target_est_x"] - df[mocap_target_x]), label="tf est x")
#     ax.plot(df.index, np.abs(df["target_est_y"] - df[mocap_target_y]), label="tf est y")
#     ax.plot(df.index, np.abs(df["target_est_z"] - df[mocap_target_z]), label="tf est z")
    
#     ax.plot(df.index, np.abs(df[mocap_gtsam_target_x] - df[mocap_target_x]), label="mocap est x")
#     ax.plot(df.index, np.abs(df[mocap_gtsam_target_y] - df[mocap_target_y]), label="mocap est y")
#     ax.plot(df.index, np.abs(df[mocap_gtsam_target_z] - df[mocap_target_z]), label="mocap est z")
    
#     ax.plot(df.index, np.linalg.norm([df["target_est_x"] - df[mocap_target_x],
#                                        df["target_est_y"] - df[mocap_target_y],
#                                        df["target_est_z"] - df[mocap_target_z]], axis=0), label="est norm")
                  
#     ax.plot(df.index, np.linalg.norm([df[mocap_gtsam_target_x] - df[mocap_target_x],
#                                        df[mocap_gtsam_target_y] - df[mocap_target_y],
#                                        df[mocap_gtsam_target_z] - df[mocap_target_z]], axis=0), label="mocap norm")
#     ax.plot(df.index, np.abs(df[mocap_gtsam_target_y] - df[mocap_target_y]), label="mocap est y")
#     ax.plot(df.index, np.abs(df[mocap_gtsam_target_z] - df[mocap_target_z]), label="mocap est z")

    ax.plot(dfs[0].index, dfs[0][mocap_drone_x], label="drone mocap x")
    ax.plot(dfs[0].index, dfs[0][mocap_drone_y], label="drone mocap y")
    ax.plot(dfs[0].index, dfs[0][mocap_drone_z], label="drone mocap z")

    # ax.plot(dfs[0].index, dfs[0][mavros_in_x], label="drone t265 x")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_y], label="drone t265 y")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_z], label="drone t265 z")

    # ax.plot(dfs[0].index, dfs[0]['t265_wrt_mocap_x'], label="t265 wrt mocap x")
    # ax.plot(dfs[0].index, dfs[0]['t265_wrt_mocap_y'], label="t265 wrt mocap y")
    # ax.plot(dfs[0].index, dfs[0]['t265_wrt_mocap_z'], label="t265 wrt mocap z")

    # ax.plot(dfs[0].index, dfs[0][mavros_in_x] + dfs[0][mocap_drone_x], label="sum x")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_y] + dfs[0][mocap_drone_y], label="sum y")
    # ax.plot(dfs[0].index, dfs[0][mavros_in_z] + dfs[0][mocap_drone_z], label="sum z")

    ax.set_xlim([-30,0])

#     ax.set_ylim([0,.2])
    ax.legend()


# + active=""
# print(mavros_wrt_mocap)
# -

#

for rosbag, ulog in zip(rosbags, ulogs):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
#     ax.plot(ulog["commanded"].index, ulog["commanded"]["current.type"], label="Ulog x")
#     ax.plot(ulog["visual_odometry"].index, -ulog["visual_odometry"]["x"], label="Ulog x")
    ax.plot(ulog["visual_odometry"].index, -ulog["visual_odometry_y"], label="Ulog y")
#     ax.plot(ulog["visual_odometry"].index, -ulog["visual_odometry"]["z"], label="-Ulog z")
#     ax.plot(ulog["position"].index, -ulog["position"]["x"], label="Ulog pos x")
#     ax.plot(ulog["position"].index, -ulog["position"]["y"], label="Ulog pos y")
#     ax.plot(ulog["position"].index, -ulog["position"]["z"], label="-Ulog pos z")
    ax.plot(rosbag.index, rosbag["mavros-odometry-out-pose.pose.position.x"], label="Bag x")
#     ax.plot(rosbag.index, rosbag["mavros-odometry-out-pose.pose.position.y"], label="Bag y")
#     ax.plot(rosbag.index, rosbag["mavros-odometry-out-pose.pose.position.z"], label="Bag z")
#     ax.plot(rosbag.index, rosbag["grasp_state_machine_node-grasp_started-data"], label="Bag z")
#     ax.set_xlim(right=25)
    ax.legend()



print(rosbags[0].index)


# +
def reindex_df(df, dt="100ms"):
    all_int_cols = ["current.type", "grasp_state_machine_node-grasp_started-data"]
    all_cols = df.columns
    float_cols = [x for x in all_cols if x not in all_int_cols]
    int_cols = [x for x in all_cols if x in all_int_cols]
    df.set_index('times', inplace=True, drop=False)
    df.index = pd.to_timedelta(df.index, unit='s')
#     print("Before reindex", df.index)
    times = pd.timedelta_range(start='0s', freq=dt, periods=len(df))
    df = df.reindex(df.index.union(times))
#     print("Reindexed", df.index)
#     print("Reindexed", df["mavros-odometry-out-pose.pose.position.x"])
    df.loc[:,float_cols] = df.loc[:,float_cols].interpolate(method='time')
    
    df.loc[:,int_cols] = df.loc[:,int_cols].ffill()
    df = df.reindex(times)
#     print(df.index)
    return df

def recenter_df(df, start_time):
    start_time = pd.to_timedelta(start_time, unit='s')
    idx = df.index.get_loc(start_time, method='nearest')
    nearest_time = df.iloc[[idx]].index
    delta = nearest_time.to_pytimedelta()
    df.index = (df.index - delta).total_seconds()
    return df

def get_rosbag_trajectory_start(rosbag):
    return rosbag["grasp_state_machine_node-grasp_started-data"].dropna().index[0]

aligned_pos_dfs = []
aligned_cmd_dfs = []
aligned_vo_dfs = []
for ulog in ulogs:
    aligned_pos_df = reindex_df(ulog["position"])
    aligned_pos_dfs.append(aligned_pos_df)
    aligned_cmd_df = reindex_df(ulog["commanded"], dt="200ms")
    aligned_cmd_dfs.append(aligned_cmd_df)
    aligned_vo_df = reindex_df(ulog["visual_odometry"])
    aligned_vo_dfs.append(aligned_vo_df)
    
    start_time, end_time = get_trajectory_bounds({"commanded": aligned_cmd_df}, 10, start_padding=0.0)
    recenter_df(aligned_pos_df, start_time)
    recenter_df(aligned_vo_df, start_time)
    
aligned_rosbag_dfs = []
import datetime
for i, rosbag in enumerate(rosbags):
    rosbag.index = pd.to_timedelta(rosbag.index.strftime('%H:%M:%S.%f'))
    delta = rosbag.iloc[[0]].index.to_pytimedelta()
    rosbag.index = (rosbag.index - delta).total_seconds()
    rosbag["times"] = rosbag.index
    aligned_rosbag_df = reindex_df(rosbag, dt="100ms")
    orig_index = aligned_rosbag_df.index.copy()
    print(orig_index)
    rosbag_start_time = get_rosbag_trajectory_start(aligned_rosbag_df)
#     print("Bag start", rosbag_start_time)
#     aligned_rosbag_df.index = (aligned_rosbag_df.index - rosbag_start_time).total_seconds()
#     print(aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"])
    error = 1
    last_error = np.inf
    count = 1
    while error <= last_error:
        aligned_rosbag_df.index = (aligned_rosbag_df.index).total_seconds()
        print("------------------------------")
        if count > 1:
            last_error = error
        error = np.linalg.norm(-aligned_vo_dfs[i]["y"].values[:len(aligned_rosbag_df)] - aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"].values)
#         error = np.mean(- aligned_vo_df["x"] - aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"])
#         print("Error", error)
#         print("Last Error", last_error)
#         print("REF", -aligned_vo_dfs[i]["y"].values[500])
#         print("CUR", aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"].values[500])
#         print(aligned_rosbag_df.index)
        aligned_rosbag_df.index =  (aligned_rosbag_df.index - 0.005 * count)
        aligned_rosbag_df["times"] = aligned_rosbag_df.index
#         print(aligned_rosbag_df.index)
#         print("Before", aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"])
        aligned_rosbag_df = reindex_df(aligned_rosbag_df, dt="100ms")
        
#         print("After", aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"])
        count += 1
    print("Shifted ", 0.005* count)
#         break
#         aligned_rosbag_df.index = aligned_rosbag_df.index.total_seconds()
    aligned_rosbag_df.index = (orig_index - rosbag_start_time).total_seconds() - 0.005 * (count -1)
    aligned_rosbag_dfs.append(aligned_rosbag_df)


# -

aligned_cmd_dfs[0].index

np.linalg.norm(-aligned_vo_df["x"].values[:len(aligned_rosbag_df)] - aligned_rosbag_df["mavros-odometry-out-pose.pose.position.x"].values)

print(-aligned_vo_dfs[0]["y"])
# print(aligned_rosbag_dfs[0]["mavros-odometry-out-pose.pose.position.x"].values[250])

# +
for ulog, bag in zip(aligned_vo_dfs, aligned_rosbag_dfs):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    
#     times = np.array(ulog.index)
#     positions = np.array([ulog["x"], ulog["y"], ulog["z"]]).T
#     quaternions_wxyz = np.array([ulog["q[0]"], ulog["q[1]"], ulog["q[2]"], ulog["q[3]"]]).T
#     ulog_evo = convert_traj_to_evo(positions, quaternions_wxyz, times)

#     times = np.array(bag.index)
#     positions = np.array([bag["mavros-odometry-out-pose.pose.position.x"], 
#                           bag["mavros-odometry-out-pose.pose.position.y"],
#                           bag["mavros-odometry-out-pose.pose.position.z"]]).T
#     quaternions_wxyz = np.array([bag["mavros-odometry-out-pose.pose.orientation.w"],
#                                  bag["mavros-odometry-out-pose.pose.orientation.x"], 
#                                  bag["mavros-odometry-out-pose.pose.orientation.y"], 
#                                  bag["mavros-odometry-out-pose.pose.orientation.z"]]).T
#     bag_evo = convert_traj_to_evo(positions, quaternions_wxyz, times)

#     T, (bag_synced, ulog_synced) = get_aligned_trajectories(ulog_evo, bag_evo)
    
#     ax.plot(ulog_synced.timestamps, ulog_synced.positions_xyz[:,0], label="Ulog x")
#     ax.plot(ulog_synced.timestamps, ulog_synced.positions_xyz[:,1], label="Ulog y")
#     ax.plot(ulog_synced.timestamps, ulog_synced.positions_xyz[:,2], label="Ulog z")
#     ax.plot(bag_synced.timestamps, bag_synced.positions_xyz[:,0], label="Bag x")
#     ax.plot(bag_synced.timestamps, bag_synced.positions_xyz[:,1], label="Bag y")
#     ax.plot(bag_synced.timestamps, bag_synced.positions_xyz[:,2], label="Bag z")

#     print(T)
#     print(ulog_synced.positions_xyz[:,0])
    ax.plot(ulog.index, -ulog["x"], label="Ulog x")
    ax.plot(ulog.index, -ulog["y"], label="Ulog y")
    ax.plot(ulog.index, -ulog["z"], label="-Ulog z")
    ax.plot(bag.index, bag["mavros-odometry-out-pose.pose.position.x"], label="Bag x")
    ax.plot(bag.index, bag["mavros-odometry-out-pose.pose.position.y"], label="Bag y")
    ax.plot(bag.index, bag["mavros-odometry-out-pose.pose.position.z"], label="Bag z")
    ax.set_xlim(right=25)
    ax.legend()
    
# -

for col in bag.columns:
    print(col)
    

aligned_rosbag_df["grasp_state_machine_node-grasp_started-data"].dropna()

ulogs[0]["commanded"].plot(y=["current.type"])

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

# # Vision Medkit 0.5 m/s

ulog_location = "../vision_pepsi_2ms"
ulog_result_location = "../vision_pepsi_2ms"
bag_location = "../vision_medkit_05ms"
bag_result_location = "../vision_medkit_05ms"
rosbags = load_rosbags(bag_result_location, bag_location)

# +

fig, ax = plt.subplots(1, 1, figsize=(15,5))
for rosbag in rosbags:
    rosbag["times"] = [0.01 * i for i in range(len(rosbag))]
    rosbag["start"] = rosbag["grasp_state_machine_node-grasp_started-data"] - 4
    rosbag["target_error"] = np.linalg.norm([rosbag["gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.x"] \
                                             - rosbag["sparkgrasptar-world-pose.position.x"], \
                                             rosbag["gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.y"] \
                                             - rosbag["sparkgrasptar-world-pose.position.y"], \
                                             rosbag["gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.position.z"] \
                                             - rosbag["sparkgrasptar-world-pose.position.z"]], axis=0)
    
    rosbag["t265_to_mocap_x"] = rosbag["mavros-odometry-in-pose.pose.position.x"] + rosbag["sparksdrone-world-pose.position.x"].dropna()[0]
    rosbag["t265_to_mocap_y"] = rosbag["mavros-odometry-in-pose.pose.position.y"] + rosbag["sparksdrone-world-pose.position.y"].dropna()[1]
    rosbag["t265_to_mocap_z"] = rosbag["mavros-odometry-in-pose.pose.position.z"] + rosbag["sparksdrone-world-pose.position.z"].dropna()[2]
    
    rosbag["vio_error"] = np.linalg.norm([rosbag["t265_to_mocap_x"] \
                                             - rosbag["sparksdrone-world-pose.position.x"], \
                                             rosbag["t265_to_mocap_y"] \
                                             - rosbag["sparksdrone-world-pose.position.y"], \
                                             rosbag["t265_to_mocap_z"] \
                                             - rosbag["sparksdrone-world-pose.position.z"]], axis=0)
    
    rosbag["tracking_error"] = np.linalg.norm([-rosbag["mavros-odometry-out-pose.pose.position.x"] \
                                             - rosbag["mavros-setpoint_raw-local-position.x"], \
                                             -rosbag["mavros-odometry-out-pose.pose.position.y"] \
                                             - rosbag["mavros-setpoint_raw-local-position.y"], \
                                             rosbag["mavros-odometry-out-pose.pose.position.z"] \
                                             - rosbag["mavros-setpoint_raw-local-position.z"]], axis=0)
    
    
    
    rosbag["aligned_mavros_y"] = -rosbag["mavros-odometry-out-pose.pose.position.y"]

    rosbag.plot(x="times", y=["gtsam_tracker_node_secondary-teaser_global-pose.position.x",
                             "gtsam_tracker_node_secondary-teaser_global-pose.position.y",
                             "gtsam_tracker_node_secondary-teaser_global-pose.position.z"], ax=ax)
    
fig, ax = plt.subplots(1, 1, figsize=(15,5))
# -

mean_df, std_df = get_aggregate_df(rosbags)
lower_df = mean_df - std_df
upper_df = mean_df + std_df

rosbags[0]["gtsam_tracker_node_secondary-teaser_global-pose.position.x"]
# rosbags[0]["sparkgrasptar-world-pose.position.x"]

# +
fig, ax = plt.subplots(1, 1, figsize=(15,5))
error = np.linalg.norm([rosbags[0]["gtsam_tracker_node_secondary-teaser_global-pose.position.x"] \
                                             - rosbags[0]["sparkgrasptar-world-pose.position.x"], \
                                             rosbags[0]["gtsam_tracker_node_secondary-teaser_global-pose.position.y"] \
                                             - rosbags[0]["sparkgrasptar-world-pose.position.y"], \
                                             rosbags[0]["gtsam_tracker_node_secondary-teaser_global-pose.position.z"] \
                                             - rosbags[0]["sparkgrasptar-world-pose.position.z"]], axis=0)

# ax.plot(rosbags[0]["times"], error)
p1 = ax.plot(mean_df["times"], mean_df["target_error"], label="Target Error")
p2 = ax.fill_between(mean_df["times"], lower_df["target_error"], upper_df["target_error"], alpha=0.5, linewidth=0)
# plot_std(ax, mean_df, lower_df, upper_df, ["target_error"])

p3 = ax.plot(mean_df["times"], mean_df["vio_error"], label="VIO Error")
# plot_std(ax, mean_df, lower_df, upper_df, ["vio_error"])
p4 = ax.fill_between(mean_df["times"], lower_df["vio_error"], upper_df["vio_error"], alpha=0.5, linewidth=0)

p5 = ax.plot(mean_df["times"], mean_df["tracking_error"], label="Tracking Error")
p6 = ax.fill_between(mean_df["times"], lower_df["tracking_error"], upper_df["tracking_error"], alpha=0.5, linewidth=0)

ax.plot(mean_df["times"], mean_df["grasp_state_machine_node-grasp_started-data"] - 0.5)
# plot_std(ax, mean_df, lower_df, upper_df, ["grasp_state_machine_node-grasp_started-data"])


ax.set_xlim(10,60)
ax.set_ylim(0,0.2)
# ax.legend(["Target Error", "VIO Error", "Tracking Error"])
ax.legend()
# -

for col in mean_df.columns:
    print(col)

mean_df.plot(x="times", y=["mavros-odometry-out-pose.pose.position.z",
                             "mavros-setpoint_raw-local-position.z"])


