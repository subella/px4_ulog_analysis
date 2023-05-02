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

alignment_error_x = "alignment_error_x"
alignment_error_y = "alignment_error_y"
alignment_error_z = "alignment_error_z"
alignment_error_norm = "alignment_error_norm"


mavros_gtsam_target_x = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x"
mavros_gtsam_target_y = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y"
mavros_gtsam_target_z = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z"

target_error_x = "target_error_x"
target_error_y = "target_error_y"
target_error_z = "target_error_z"
target_error_norm = "target_error_norm"


tracking_error_x = "tracking_error_x"
tracking_error_y = "tracking_error_y"
tracking_error_z = "tracking_error_z"
tracking_error_norm = "tracking_error_norm"

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
    print(rosbags[0].index)
    
#     ulogs = load_ulogs(ulog_result_location, ulog_location)
#     format_ulogs(ulogs)
    
#     refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic="mavros-odometry-out-pose.pose.position.x")

#     dfs = merge_ulogs_rosbags(ulogs, rosbags)
    return rosbags
    return dfs


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

def align_mavros_to_mocap(df, start_time=0, end_time=1):
    # Start alignment at grasp start before grasp start
    df_sliced = df.loc[start_time:]
    times = np.array(df_sliced.index)
    mavros_positions = np.array([df_sliced[mavros_in_x], 
                                 df_sliced[mavros_in_y],
                                 df_sliced[mavros_in_z]]).T
    mavros_quaternions_wxyz = np.array([df_sliced[mavros_in_qw],
                                        df_sliced[mavros_in_qx], 
                                        df_sliced[mavros_in_qy], 
                                        df_sliced[mavros_in_qz]]).T

    mocap_positions = np.array([df_sliced[mocap_drone_x], 
                                df_sliced[mocap_drone_y],
                                df_sliced[mocap_drone_z]]).T
    mocap_quaternions_wxyz = np.array([df_sliced[mocap_drone_qw],
                                       df_sliced[mocap_drone_x], 
                                       df_sliced[mocap_drone_y], 
                                       df_sliced[mocap_drone_z]]).T
    
    
    mavros_evo = convert_traj_to_evo(mavros_positions, mavros_quaternions_wxyz, times)
    mocap_evo = convert_traj_to_evo(mocap_positions, mocap_quaternions_wxyz, times)

    T, (mocap_aligned, mavros_aligned) = get_aligned_trajectories(mavros_evo, mocap_evo, 
                                                                  num_poses=len(df_sliced[start_time:end_time]))

    return T, mocap_aligned, mavros_aligned

def create_aligned_mavros(dfs):
    for i, df in enumerate(dfs):
        start_time = -5
        end_time = 3
        T, mocap_aligned, mavros_aligned = align_mavros_to_mocap(df, start_time=start_time, end_time=end_time)
        aligned_df = pd.DataFrame()
        aligned_df.index = df[start_time:].index
        aligned_df[mavros_in_aligned_x] = mavros_aligned.positions_xyz[:,0]
        aligned_df[mavros_in_aligned_y] = mavros_aligned.positions_xyz[:,1]
        aligned_df[mavros_in_aligned_z] = mavros_aligned.positions_xyz[:,2]
        aligned_df[mavros_in_aligned_qw] = mavros_aligned.orientations_quat_wxyz[:,0]
        aligned_df[mavros_in_aligned_qx] = mavros_aligned.orientations_quat_wxyz[:,1]
        aligned_df[mavros_in_aligned_qy] = mavros_aligned.orientations_quat_wxyz[:,2]
        aligned_df[mavros_in_aligned_qz] = mavros_aligned.orientations_quat_wxyz[:,3]
        dfs[i] = df.join(aligned_df)

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

def plot_mean_std(ax, mean_df, vals, std, label=None):
    ax.plot(mean_df.index, vals, label=label)
    ax.fill_between(mean_df.index, 
                    vals - std,
                    vals + std, 
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
        
    mean_df = aggregate_df.groupby(aggregate_df.index).mean()

    std_df = aggregate_df.groupby(aggregate_df.index).std()
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

# # Vision Medkit 0.5 m/s

# +
ulog_location = "../vision_pepsi_2ms"
ulog_result_location = "../log_output"
bag_location = "../vision_pepsi_2ms"
bag_result_location = "../vision_pepsi_2ms"

dfs = create_dfs(ulog_location, ulog_result_location, bag_location, bag_result_location)
print(dfs[0])
create_aligned_mavros(dfs)

# -

print(dfs[0])


# +
def add_error_columns(df, topics_1, topics_2, error_topics, norm_name):

    df.loc[:,error_topics] = df.loc[:,topics_1] - df.loc[:,topics_2]
    for i, topic in enumerate(error_topics):
        df[topic] = df.loc[:,topics_1[i]] - df.loc[:,topics_2[i]]
    df[norm_name] = np.linalg.norm((df[error_topics[0]], df[error_topics[1]], df[error_topics[2]]),axis=0)
    return df

def verify_alignment(dfs):
    for df in dfs:
        fig, ax = plt.subplots(1, 1, figsize=(15,10))
        ax.plot(df.index, df[mocap_drone_x], label="Mocap Drone x")
        ax.plot(df.index, df[mocap_drone_y], label="Mocap Drone y")
        ax.plot(df.index, df[mocap_drone_z], label="Mocap Drone z")

        ax.plot(df.index, df[mavros_in_aligned_x], label="Aligned Marvros x")
        ax.plot(df.index, df[mavros_in_aligned_y], label="Aligned Marvros y")
        ax.plot(df.index, df[mavros_in_aligned_z], label="Aligned Marvros z")
        
verify_alignment(dfs)
for df in dfs:
    df = add_error_columns(df, [mavros_in_aligned_x, mavros_in_aligned_y, mavros_in_aligned_z],
                  [mocap_drone_x, mocap_drone_y, mocap_drone_z],
                  [alignment_error_x, alignment_error_y, alignment_error_z],
                   alignment_error_norm
                  )
    
    df = add_error_columns(df, 
                  [mocap_gtsam_target_x, mocap_gtsam_target_y, mocap_gtsam_target_z],
                  [mocap_target_x, mocap_target_y, mocap_target_z],
                  [target_error_x, target_error_y, target_error_z],
                           target_error_norm
                  )
    
    df = add_error_columns(df, 
              [mavros_in_x, mavros_in_y, mavros_in_z],
              [mavros_drone_sp_x, mavros_drone_sp_y, mavros_drone_sp_z],
              [tracking_error_x, tracking_error_y, tracking_error_z],
                       tracking_error_norm
              )
        
mean_df, std_df = get_aggregate_df(dfs) 
# -

dfs[-1][alignment_error_x].dropna()
print(mean_df)

# +
fig, ax = plt.subplots(1,1, figsize=(20,10))
# ax.plot(dfs[2][mavros_in_aligned_x])
# ax.plot(dfs[2][mocap_drone_x])
# for df in dfs:
#     ax.plot(df[mavros_in_aligned_x])
    
# ax.plot(mean_df[mavros_in_aligned_x])
ax.plot(mean_df[mavros_in_aligned_x] - mean_df[mocap_drone_x])

print(mean_df[mavros_in_aligned_x])


# +
def get_yaw(df, quat_topics):
    pass
    

def plot_composite_errors(df, ax):
    pass

# fig, ax = plt.subplots(1,1, figsize=(20,10))
# ax[0][0].plot(mean_df.index, mean_df["alignment_error_norm"])
# for df in dfs:
#     ax[0][0].plot(np.linalg.norm(mean_df[alignment_error_x],
#                                  mean_df[alignment_error_y],
#                                  mean_df[alignment_error_z]))
    
    
# ax[0][0].plot(mean_df.index, mean_df[gripper_state])    
# for df in dfs:
#     ax[0][1].set_aspect('equal', adjustable='box')
#     ax[0][1].plot(df[-15:0][mocap_drone_x], df[-15:0][mocap_drone_y])
#     ax[0][1].arrow(np.mean(df[mocap_target_x]), np.mean(df[mocap_target_y]), .1,.1)

fig, ax = plt.subplots(1,1, figsize=(20,10))
plot_mean_std(ax, mean_df, mean_df[alignment_error_norm], std_df[alignment_error_norm], label="VIO Drift Error")
plot_mean_std(ax, mean_df, mean_df[target_error_norm], std_df[target_error_norm], label="Target Error")
plot_mean_std(ax, mean_df, mean_df[tracking_error_norm], std_df[tracking_error_norm], label="Tracking Error")
ax.vlines(x=[0], ymin=0, ymax=.25, colors='0.5', ls='--', lw=2)
ax.vlines(x=[df[df[gripper_state] == 0].index[0]], ymin=0, ymax=.25, colors='0', ls='--', lw=2)
ax.set_xlim(-5, 15)
ax.set_ylim(0, 0.25)
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(20,10))
plot_mean_std(ax, mean_df, np.abs(mean_df[alignment_error_x]), np.abs(std_df[alignment_error_x]), label="VIO Drift Error x")
plot_mean_std(ax, mean_df, np.abs(mean_df[target_error_x]), np.abs(std_df[target_error_x]), label="Target Error x")
plot_mean_std(ax, mean_df, np.abs(mean_df[tracking_error_x]), np.abs(std_df[tracking_error_x]), label="Tracking Error x")
ax.vlines(x=[0], ymin=0, ymax=.25, colors='0.5', ls='--', lw=2)
ax.vlines(x=[df[df[gripper_state] == 0].index[0]], ymin=0, ymax=.25, colors='0', ls='--', lw=2)
ax.set_xlim(-5, 15)
ax.set_ylim(0, 0.25)
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(20,10))
plot_mean_std(ax, mean_df, np.abs(mean_df[alignment_error_y]), np.abs(std_df[alignment_error_y]), label="VIO Drift Error y")
plot_mean_std(ax, mean_df, np.abs(mean_df[target_error_y]), std_df[target_error_y], label="Target Error y")
plot_mean_std(ax, mean_df, np.abs(mean_df[tracking_error_y]), std_df[tracking_error_y], label="Tracking Error y")
ax.vlines(x=[0], ymin=0, ymax=.25, colors='0.5', ls='--', lw=2)
ax.vlines(x=[df[df[gripper_state] == 0].index[0]], ymin=0, ymax=.25, colors='0', ls='--', lw=2)
ax.set_xlim(-5, 15)
ax.set_ylim(0, 0.25)
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(20,10))
plot_mean_std(ax, mean_df, np.abs(mean_df[alignment_error_z]), std_df[alignment_error_z], label="VIO Drift Error z")
plot_mean_std(ax, mean_df, np.abs(mean_df[target_error_z]), std_df[target_error_z], label="Target Error z")
plot_mean_std(ax, mean_df, np.abs(mean_df[tracking_error_z]), std_df[tracking_error_z], label="Tracking Error z")
ax.vlines(x=[0], ymin=0, ymax=.25, colors='0.5', ls='--', lw=2)
ax.vlines(x=[df[df[gripper_state] == 0].index[0]], ymin=0, ymax=.25, colors='0', ls='--', lw=2)
ax.set_xlim(-5, 15)
ax.set_ylim(0, 0.25)
ax.legend()

# plot_mean_std(ax[1][0], mean_df, std_df, "alignment_error_x", label="VIO Drift Error x")
# plot_mean_std(ax[1][0], mean_df, std_df, "target_error_x", label="Target Error x")
# plot_mean_std(ax[1][0], mean_df, std_df, "tracking_error_x", label="Tracking Error x")

# plot_mean_std(ax[2][0], mean_df, std_df, "alignment_error_y", label="VIO Drift Error y")
# plot_mean_std(ax[2][0], mean_df, std_df, "target_error_y", label="Target Error y")
# plot_mean_std(ax[2][0], mean_df, std_df, "tracking_error_y", label="Tracking Error y")

# plot_mean_std(ax[3][0], mean_df, std_df, "alignment_error_z", label="VIO Drift Error z")
# plot_mean_std(ax[3][0], mean_df, std_df, "target_error_z", label="Target Error z")
# plot_mean_std(ax[3][0], mean_df, std_df, "tracking_error_z", label="Tracking Error z")

# for sub_ax in ax:
#     sub_ax[0].vlines(x=[0], ymin=0, ymax=.25, colors='0.5', ls='--', lw=2)
#     sub_ax[0].vlines(x=[df[df[gripper_state] == 0].index[0]], ymin=0, ymax=.25, colors='0', ls='--', lw=2)
#     sub_ax[0].set_xlim(-5, 15)
#     sub_ax[0].set_ylim(0, 0.25)
#     sub_ax[0].legend()

# for df in dfs:
#     ax[1][0].plot(df.index, df[mocap_drone_x])
#     ax[1][0].plot(df.index, df[mocap_drone_y])
#     ax[1][0].plot(df.index, df[mocap_drone_z])
    
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_drone_x, label="Mocap Drone x")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_drone_y, label="Mocap Drone y")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_drone_z, label="Mocap Drone z")
    
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_target_x, label="Mocap Target x")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_target_y, label="Mocap Target y")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_target_z, label="Mocap Target z")

# plot_mean_std(ax[1][0], mean_df, std_df, mocap_gtsam_target_x, label="Vision Target x")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_gtsam_target_y, label="Vision Target y")
# plot_mean_std(ax[1][0], mean_df, std_df, mocap_gtsam_target_z, label="Vision Target z")
    

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


