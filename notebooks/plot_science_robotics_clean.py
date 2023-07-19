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
import matplotlib.gridspec as gridspec

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
mocap_target_qx = "sparkgrasptar-world-pose.orientation.x"
mocap_target_qy = "sparkgrasptar-world-pose.orientation.y"
mocap_target_qz = "sparkgrasptar-world-pose.orientation.z"
mocap_target_qw = "sparkgrasptar-world-pose.orientation.w"

mocap_target_vx = "sparkgrasptar-world-pose.velocity.x"
mocap_target_vy = "sparkgrasptar-world-pose.velocity.y"
mocap_target_vz = "sparkgrasptar-world-pose.velocity.z"

mocap_drone_x = "sparksdrone-world-pose.position.x"
mocap_drone_y = "sparksdrone-world-pose.position.y"
mocap_drone_z = "sparksdrone-world-pose.position.z"
mocap_drone_qw = "sparksdrone-world-pose.orientation.w"
mocap_drone_qx = "sparksdrone-world-pose.orientation.x"
mocap_drone_qy = "sparksdrone-world-pose.orientation.y"
mocap_drone_qz = "sparksdrone-world-pose.orientation.z"

mocap_drone_vx = "sparksdrone-world-pose.velocity.x"
mocap_drone_vy = "sparksdrone-world-pose.velocity.y"
mocap_drone_vz = "sparksdrone-world-pose.velocity.z"

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
mavros_gtsam_target_qx = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.x"
mavros_gtsam_target_qy = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.y"
mavros_gtsam_target_qz = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.z"
mavros_gtsam_target_qw = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.w"

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
mocap_gtsam_target_qx = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.orientation.x"
mocap_gtsam_target_qy = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.orientation.y"
mocap_gtsam_target_qz = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.orientation.z"
mocap_gtsam_target_qw = "gtsam_tracker_node_secondary-target_global_odom_estimate-pose.pose.orientation.w"



ulog_vo_x = "ulog_visual_odometry_x"
ulog_vo_y = "ulog_visual_odometry_y"
ulog_vo_z = "ulog_visual_odometry_z"
ulog_vo_qw = "ulog_visual_odometry_q[0]"
ulog_vo_qx = "ulog_visual_odometry_q[1]"
ulog_vo_qy = "ulog_visual_odometry_q[2]"
ulog_vo_qz = "ulog_visual_odometry_q[3]"

ulog_sp_x = "ulog_position_sp_x"
ulog_sp_y = "ulog_position_sp_y"
ulog_sp_z = "ulog_position_sp_z"
ulog_sp_vx = "ulog_position_sp_vx"
ulog_sp_vy = "ulog_position_sp_vy"
ulog_sp_vz = "ulog_position_sp_vz"

ulog_x = "ulog_position_x"
ulog_y = "ulog_position_y"
ulog_z = "ulog_position_z"
ulog_vx = "ulog_position_vx"
ulog_vy = "ulog_position_vy"
ulog_vz = "ulog_position_vz"

ulog_error_x = "ulog_error_x"
ulog_error_y = "ulog_error_y"
ulog_error_z = "ulog_error_z"
ulog_error_vx = "ulog_error_vx"
ulog_error_vy = "ulog_error_vy"
ulog_error_vz = "ulog_error_vz"
ulog_pos_error_norm = "ulog_pos_error_norm"
ulog_vel_error_norm = "ulog_vel_error_norm"

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
        format_ulog_index(ulog["position_sp"])
        # First row of commanded is corrupted
        ulog["commanded"].drop([0])
        format_ulog_index(ulog["commanded"])
        format_ulog_index(ulog["visual_odometry"])
        
def reindex_ulogs(ulogs):
    for i in range(len(ulogs)):
        ulogs[i]["position"] = reindex_df(ulogs[i]["position"])
        ulogs[i]["position_sp"] = reindex_df(ulogs[i]["position_sp"])
        ulogs[i]["commanded"] = reindex_df(ulogs[i]["commanded"], int_cols=["current.type"])
        ulogs[i]["visual_odometry"] = reindex_df(ulogs[i]["visual_odometry"])
        
def get_ulog_grasp_start_time(df):
    """Get the trajectory time range if possible."""
    return df[df["current.type"] == 8].index[0]

def center_ulog(ulog):
    start_time = get_ulog_grasp_start_time(ulog["commanded"])
    ulog["position"].index -= start_time
    ulog["position_sp"].index -= start_time
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
        print(b.topics)
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
        df = load_bag(result_dir, file, bool_topics=["grasp_state_machine_node-grasp_started",
                                                     "cmd_gripper_sub"])
        dfs.append(df)
    return dfs

def format_rosbag_index(rosbag, sample_rate=10):
    """Converts indexes to time as float, starting at 0"""
    rosbag.index = pd.to_timedelta(rosbag.index.strftime('%H:%M:%S.%f'))
    rosbag.index = rosbag.index.total_seconds()
    
def format_rosbags_index(rosbags, sample_rate=10):
    for rosbag in rosbags:
        rosbag = format_rosbag_index(rosbag, sample_rate)
        
def reindex_rosbags(rosbags, sample_rate=10):
    for i in range(len(rosbags)):
        rosbags[i] = reindex_df(rosbags[i], 
                                int_cols=[grasp_segment, gripper_state])
 
# def get_rosbag_grasp_time(rosbag):
#     print(rosbag["cmd_gripper_sub-data"])
       
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
    ulog["position_sp"] = ulog["position_sp"].add_prefix("ulog_position_sp_")
    ulog["commanded"] = ulog["commanded"].add_prefix("ulog_commanded_")
    ulog["visual_odometry"] = ulog["visual_odometry"].add_prefix("ulog_visual_odometry_")
    
    rosbag.index = pd.to_timedelta(rosbag.index, unit='s')
    ulog["position"].index = pd.to_timedelta(ulog["position"].index, unit='s')
    ulog["position_sp"].index = pd.to_timedelta(ulog["position_sp"].index, unit='s')
    ulog["commanded"].index = pd.to_timedelta(ulog["commanded"].index, unit='s')
    ulog["visual_odometry"].index = pd.to_timedelta(ulog["visual_odometry"].index, unit='s')
    
    df = pd.merge(rosbag, ulog["position"], left_index=True, right_index=True)
    df = pd.merge(df, ulog["position_sp"], left_index=True, right_index=True)
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
              sample_rate=0.1,
              alignment_topic=mocap_drone_x):
    
    rosbags = load_rosbags(bag_result_location, bag_location)
    format_rosbags(rosbags, sample_rate=10)
    
    if ulog_location is None:
        return rosbags
    
    ulogs = load_ulogs(ulog_result_location, ulog_location)
    format_ulogs(ulogs)
    
    refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic=alignment_topic)

    dfs = merge_ulogs_rosbags(ulogs, rosbags)
    return dfs

def validate_alignment(dfs, alignment_topic):
    for df in dfs:
        fig, ax = plt.subplots(1,1)
        ax.plot(df.index, df[alignment_topic], label="Bag")
        ax.plot(df.index, -df[ulog_vo_y], label="Ulog")
        ax.legend()


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
        end_time = 4
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
#     rosbag.index += .2
#     rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
    step = 0.05
    shift = 0
    size = 500
    last_error = np.inf
    error = -np.inf
    while error < last_error:
        print("error", error)
        
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
#     df["sparkgrasptar-world-pose.velocity.x"] = (df["sparkgrasptar-world-pose.position.x"].diff().rolling(20).mean()) / dt
#     df["sparkgrasptar-world-pose.velocity.y"] = (df["sparkgrasptar-world-pose.position.y"].diff().rolling(20).mean()) / dt
#     df["sparkgrasptar-world-pose.velocity.z"] = (df["sparkgrasptar-world-pose.position.z"].diff().rolling(20).mean()) / dt

def add_velocities(dfs):
    for df in dfs:
        add_velocity(df)
    

# -

# # Static Plots

# Load all the data

# +
# ulog_location = "../vision_medkit_05ms"
# ulog_result_location = "../log_output"
# bag_location = "../vision_medkit_05ms"
# bag_result_location = "../vision_medkit_05ms"

# medkit_dfs = create_dfs(ulog_location, ulog_result_location, 
#                         bag_location, bag_result_location, 
#                         alignment_topic=mavros_out_x)
# validate_alignment(medkit_dfs, mavros_out_x)
# create_aligned_mavros(medkit_dfs)

# ulog_location = "../vision_cardboard_box_05ms"
# ulog_result_location = "../log_output"
# bag_location = "../vision_cardboard_box_05ms"
# bag_result_location = "../vision_cardboard_box_05ms"

# cardboard_box_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mavros_out_x)
# validate_alignment(cardboard_box_dfs, mavros_out_x)
# create_aligned_mavros(cardboard_box_dfs)

# ulog_location = "../vision_pepsi_05ms"
# ulog_result_location = "../log_output"
# bag_location = "../vision_pepsi_05ms"
# bag_result_location = "../vision_pepsi_05ms"

# pepsi_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mavros_out_x)
# validate_alignment(pepsi_dfs, mavros_out_x)
# create_aligned_mavros(pepsi_dfs)

# ulog_location = "../vision_pepsi_125ms"
# ulog_result_location = "../log_output"
# bag_location = "../vision_pepsi_125ms"
# bag_result_location = "../vision_pepsi_125ms"

# pepsi_mid_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mavros_out_x)
# validate_alignment(pepsi_mid_dfs, mavros_out_x)
# create_aligned_mavros(pepsi_mid_dfs)

# ulog_location = "../vision_pepsi_2ms"
# ulog_result_location = "../log_output"
# bag_location = "../vision_pepsi_2ms"
# bag_result_location = "../vision_pepsi_2ms"

# pepsi_fast_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mavros_out_x)
# validate_alignment(pepsi_fast_dfs, mavros_out_x)
# create_aligned_mavros(pepsi_fast_dfs)

ulog_location = None
ulog_result_location = None
bag_location = "../vision_pepsi_3ms"
bag_result_location = "../vision_pepsi_3ms"

pepsi_fastest_dfs = create_dfs(ulog_location, ulog_result_location, 
                               bag_location, bag_result_location)
create_aligned_mavros(pepsi_fastest_dfs)
# -

medkit_dfs[0][gripper_state].plot()

# # Vision Medkit 0.5 m/s

# +
ulog_location = "../vision_medkit_05ms"
ulog_result_location = "../log_output"
bag_location = "../vision_medkit_05ms"
bag_result_location = "../vision_medkit_05ms"

medkit_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(medkit_dfs, mavros_in_x)
create_aligned_mavros(medkit_dfs)

ulog_location = "../vision_cardboard_box_05ms"
ulog_result_location = "../log_output"
bag_location = "../vision_cardboard_box_05ms"
bag_result_location = "../vision_cardboard_box_05ms"

cardboard_box_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(cardboard_box_dfs, mavros_in_x)
create_aligned_mavros(cardboard_box_dfs)

ulog_location = "../vision_pepsi_05ms"
ulog_result_location = "../log_output"
bag_location = "../vision_pepsi_05ms"
bag_result_location = "../vision_pepsi_05ms"

pepsi_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(pepsi_dfs, mavros_in_x)
create_aligned_mavros(pepsi_dfs)

ulog_location = "../vision_pepsi_125ms"
ulog_result_location = "../log_output"
bag_location = "../vision_pepsi_125ms"
bag_result_location = "../vision_pepsi_125ms"

pepsi_mid_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(pepsi_mid_dfs, mavros_in_x)
create_aligned_mavros(pepsi_mid_dfs)

ulog_location = "../vision_pepsi_2ms"
ulog_result_location = "../log_output"
bag_location = "../vision_pepsi_2ms"
bag_result_location = "../vision_pepsi_2ms"

pepsi_fast_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(pepsi_fast_dfs, mavros_in_x)
create_aligned_mavros(pepsi_fast_dfs)

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
#         ax.plot(df.index, df[mocap_drone_x] - df[mavros_in_aligned_x], label="Mocap Drone x")
#         ax.plot(df.index, df[mocap_drone_y] - df[mavros_in_aligned_y], label="Mocap Drone y")
#         ax.plot(df.index, df[mocap_drone_z] - df[mavros_in_aligned_z], label="Mocap Drone z")

        ax.plot(df.index, df[mocap_drone_x], label="Mocap Drone x")
        ax.plot(df.index, df[mocap_drone_y], label="Mocap Drone y")
        ax.plot(df.index, df[mocap_drone_z], label="Mocap Drone z")
        
        ax.plot(df.index, df[mavros_in_aligned_x], label="Aligned Marvros x")
        ax.plot(df.index, df[mavros_in_aligned_y], label="Aligned Marvros y")
        ax.plot(df.index, df[mavros_in_aligned_z], label="Aligned Marvros z")
        
        
        
        ax.axvline(x=[df[df[gripper_state] == 0].index[0]], color='0.5', ls='--', lw=2)
        
#         fig, ax = plt.subplots(1, 1, figsize=(15,10))
#         ax.plot(df.index, df[mavros_in_x], label="Marvros x")
#         ax.plot(df.index, df[mavros_in_y], label="Marvros y")
#         ax.plot(df.index, df[mavros_in_z], label="Marvros z")
        
#         ax.plot(df.index, df[mavros_drone_sp_x], label="Marvros sp x")
#         ax.plot(df.index, df[mavros_drone_sp_y], label="Marvros sp y")
#         ax.plot(df.index, df[mavros_drone_sp_z], label="Marvros sp z")
#         ax.legend()
#         ax.axvline(x=[df[df[gripper_state] == 0].index[0]], color='0.5', ls='--', lw=2)    
def add_errors(dfs, vision=True):
    for df in dfs:
        if vision:
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
        df = add_error_columns(df, [ulog_x, ulog_y, ulog_z],
                      [ulog_sp_x, ulog_sp_y, ulog_sp_z],
                      [ulog_error_x, ulog_error_y, ulog_error_z],
                       ulog_pos_error_norm
                      )

        df = add_error_columns(df, [ulog_vx, ulog_vy, ulog_vz],
                      [ulog_sp_vx, ulog_sp_vy, ulog_sp_vz],
                      [ulog_error_vx, ulog_error_vy, ulog_error_vz],
                       ulog_vel_error_norm
                      )

    mean_df, std_df = get_aggregate_df(dfs) 
    return mean_df, std_df


# -

verify_alignment(pepsi_fastest_dfs)

add_velocities(pepsi_fastest_dfs)
pepsi_fastest_dfs[0].plot(y=["sparksdrone-world-pose.velocity.x"])
print(np.max(pepsi_fastest_dfs[0]["sparksdrone-world-pose.velocity.x"]))

medkit_mean_df, medkit_std_df = add_errors(medkit_dfs)
cardboard_box_mean_df, cardboard_box_std_df = add_errors(cardboard_box_dfs)
pepsi_mean_df, pepsi_std_df = add_errors(pepsi_dfs)
pepsi_mid_mean_df, pepsi_mid_std_df = add_errors(pepsi_mid_dfs)
pepsi_fast_mean_df, pepsi_fast_std_df = add_errors(pepsi_fast_dfs)

# +

sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(2,1, figsize=(6,6))
plot_mean_std(ax[0], medkit_mean_df, medkit_mean_df[ulog_pos_error_norm], medkit_std_df[ulog_pos_error_norm], label="Medkit")
plot_mean_std(ax[0], cardboard_box_mean_df, cardboard_box_mean_df[ulog_pos_error_norm], cardboard_box_std_df[ulog_pos_error_norm], label="Cardboard Box")
plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_pos_error_norm], pepsi_std_df[ulog_pos_error_norm], label="2 Liter Bottle")
# plot_mean_std(ax[0], medkit_mean_df, np.abs(medkit_mean_df[ulog_error_z]), medkit_std_df[ulog_error_z], label="Medkit")
# plot_mean_std(ax[0], cardboard_box_mean_df, np.abs(cardboard_box_mean_df[ulog_error_z]), cardboard_box_std_df[ulog_error_z], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, np.abs(pepsi_mean_df[ulog_error_z]), pepsi_std_df[ulog_error_z], label="2 Liter Bottle")

ax[0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0].axvline(x=[4], color='0', ls='--', lw=2)
ax[0].set_xlim(0, 15)
ax[0].set_ylim(0, 0.25)
ax[0].set_ylabel('Pos. Errors [m]')
# ax[0].set_xlabel('Time [s]')
ax[0].set_xticklabels([])
ax[0].legend()

plot_mean_std(ax[1], medkit_mean_df, medkit_mean_df[ulog_vel_error_norm], medkit_std_df[ulog_vel_error_norm], label="Medkit")
plot_mean_std(ax[1], cardboard_box_mean_df, cardboard_box_mean_df[ulog_vel_error_norm], cardboard_box_std_df[ulog_vel_error_norm], label="Cardboard Box")
plot_mean_std(ax[1], pepsi_mean_df, pepsi_mean_df[ulog_vel_error_norm], pepsi_std_df[ulog_vel_error_norm], label="2 Liter Bottle")
ax[1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1].set_xlim(0, 15)
ax[1].set_ylim(0, 0.5)
ax[1].set_ylabel('Vel. Errors [m/s]')
ax[1].set_xlabel('Time [s]')
ax[1].legend()

fig, ax = plt.subplots(2,1, figsize=(6,6))
plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_pos_error_norm], pepsi_std_df[ulog_pos_error_norm], label="0.5 m/s")
plot_mean_std(ax[0], pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_pos_error_norm], pepsi_mid_std_df[ulog_pos_error_norm], label="1.25 m/s")
plot_mean_std(ax[0], pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_pos_error_norm], pepsi_fast_std_df[ulog_pos_error_norm], label="2 m/s")
ax[0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0].axvline(x=[4], color='0', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_mean_df[pepsi_mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_mid_mean_df[pepsi_mid_mean_df[gripper_state] == 0].index[0]], color='0.4', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_fast_mean_df[pepsi_fast_mean_df[gripper_state] == 0].index[0]], color='0.4', ls='--', lw=2)
ax[0].set_xlim(0, 15)
ax[0].set_ylim(0, 0.5)
# ax[0].set_ylabel('Pos. Errors [m]')
# ax[0].set_xlabel('Time [s]')
ax[0].set_xticklabels([])
ax[0].legend()

plot_mean_std(ax[1], pepsi_mean_df, pepsi_mean_df[ulog_vel_error_norm], pepsi_std_df[ulog_vel_error_norm], label="0.5 m/s")
plot_mean_std(ax[1], pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_vel_error_norm], pepsi_mid_std_df[ulog_vel_error_norm], label="1.25 m/s")
plot_mean_std(ax[1], pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_vel_error_norm], pepsi_fast_std_df[ulog_vel_error_norm], label="2 m/s")
ax[1].axvline(x=[0], color='0.5', ls='--', lw=2)
# ax[1].axvline(x=[medkit_mean_df[medkit_mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
ax[1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1].set_xlim(0, 15)
ax[1].set_ylim(0, 0.6)
# ax[1].set_ylabel('Vel. Errors [m/s]')
ax[1].set_xlabel('Time [s]')
ax[1].legend()

# fig, ax = plt.subplots(2,1, figsize=(10,10))
# plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_error_x], pepsi_std_df[ulog_error_x], label="x slow")
# # plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_error_y], pepsi_std_df[ulog_error_x], label="y slow")
# plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_error_z], pepsi_std_df[ulog_error_z], label="z slow")
# plot_mean_std(ax[0], pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_x], pepsi_fast_std_df[ulog_error_x], label="x")
# # plot_mean_std(ax[0], pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_y], pepsi_fast_std_df[ulog_error_y], label="y")
# plot_mean_std(ax[0], pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_z], pepsi_fast_std_df[ulog_error_z], label="z")
# ax[0].axvline(x=[0], color='0.5', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_mean_df[pepsi_mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_mid_mean_df[pepsi_mid_mean_df[gripper_state] == 0].index[0]], color='0.4', ls='--', lw=2)
# ax[0].axvline(x=[pepsi_fast_mean_df[pepsi_fast_mean_df[gripper_state] == 0].index[0]], color='0.4', ls='--', lw=2)
# ax[0].set_xlim(0, 15)
# ax[0].set_ylim(-0.5, 0.5)
# ax[0].set_ylabel('Pos. Errors [m]')
# # ax[0].set_xlabel('Time [s]')
# ax[0].set_xticklabels([])
# ax[0].legend()

fig, ax = plt.subplots(1,1, figsize=(6,3))
N = 5
vision = (9, 6, 10, 10, 8)
mocap = (7, 8, 10, 9, 6)
ind = np.arange(N)
width = 0.3       

# Plotting
ax.bar(ind, vision , width, label='Vision')
ax.bar(ind + width, mocap, width, label='Mocap')

# plt.xlabel('Here goes x-axis label')
ax.set_ylabel('Success')
# plt.title('Here goes title of the plot')
ax.set_xticks(ind + width / 2, ('Medkit', 'Cardboard\n Box', 'Pepsi', 'Pepsi', 'Pepsi'))
# plt.xticks(rotation = 45)

ax.legend(loc='best')
ax.axhline(y=10, color='0', ls='--', lw=2)



# -

for df in (medkit_mean_df, pepsi_mean_df, cardboard_box_mean_df, pepsi_mid_mean_df, pepsi_fast_mean_df):
    planned_row = df.loc[[0]]
    # print(planned_row[mavros_gtsam_target_qx, mavros_gtsam_target_qy])
    est_quat = np.array([planned_row[mocap_gtsam_target_qx], planned_row[mocap_gtsam_target_qy],
                planned_row[mocap_gtsam_target_qz], planned_row[mocap_gtsam_target_qw]]).reshape(4,)

    gt_quat = np.array([planned_row[mocap_target_qx], planned_row[mocap_target_qy],
               planned_row[mocap_target_qz], planned_row[mocap_target_qw]]).reshape(4,)

    est_t = np.array([planned_row[mocap_gtsam_target_x], planned_row[mocap_gtsam_target_y],
                planned_row[mocap_gtsam_target_z]]).reshape(3,)
    
    gt_t = np.array([planned_row[mocap_target_x], planned_row[mocap_target_y],
            planned_row[mocap_target_z]]).reshape(3,)
    
    est_R = R.from_quat(est_quat).as_matrix()
    gt_R = R.from_quat(gt_quat).as_matrix()
    error_R = np.rad2deg(np.arccos((np.trace(est_R.T.dot(gt_R)) - 1) / 2))
    error_t = est_t - gt_t
    error_t_norm = np.linalg.norm(est_t - gt_t)
    
    proj_est_x = [est_R[0,0], est_R[0,1]] / np.linalg.norm([est_R[0,0], est_R[0,1]])
    proj_gt_x = [gt_R[0,0], gt_R[0,1]] / np.linalg.norm([gt_R[0,0], gt_R[0,1]])
    error_yaw = np.rad2deg(np.arccos((proj_est_x.dot(proj_gt_x))))
    df_sliced = df[-5:4]

    vio_error_x = np.mean(np.abs(df_sliced[alignment_error_x]))
    vio_error_y = np.mean(np.abs(df_sliced[alignment_error_y]))
    vio_error_z = np.mean(np.abs(df_sliced[alignment_error_z]))
    vio_error = [vio_error_x, vio_error_y, vio_error_z]
    vio_error_norm = np.mean(df_sliced[alignment_error_norm])
    
    print("Yaw Error", error_yaw)
    print("Rotation Error", error_R)
    print("Translation Error", error_t_norm)
    print("Translation Components", error_t)
    print("VIO Error", vio_error_norm)
    print("VIO Error Components", vio_error)
    
    print("-------------------")

# +
medkit_img = plt.imread("../vision_medkit_05ms/medkit.png")
pepsi_img = plt.imread("../vision_medkit_05ms/pepsi.png")
cardboard_img = plt.imread("../vision_medkit_05ms/cardboard.png")
# medkit_img = medkit_img[:,50:350]
# offset = 0.2
sns.set(font_scale=1.2)
def format_axes(fig):
    labels = ["A","B", "C", "D", "E", "F", "G"]
    for i, ax in enumerate(fig.axes):
        ax.text(1.0, 1.0, labels[i], va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
fig = plt.figure(figsize=(20, 16))

gs0 = gridspec.GridSpec(20, 18, figure=fig)
ax1 = fig.add_subplot(gs0[:8,:6])
ax2 = fig.add_subplot(gs0[:8,6:12])
ax3 = fig.add_subplot(gs0[:8,12:18])
err = fig.add_subplot(gs0[6:14, :12])
bar = fig.add_subplot(gs0[6:14, 12:])

plot_mean_std(err, mean_df, mean_df[alignment_error_norm], std_df[alignment_error_norm], label="VIO Drift Error")
plot_mean_std(err, mean_df, mean_df[target_error_norm], std_df[target_error_norm], label="Target Error")
plot_mean_std(err, mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="Tracking Error")
err.axvline(x=[0], color='0.5', ls='--', lw=2)
err.axvline(x=[df[df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
err.set_xlim(-5, 15)
err.set_ylim(0, 0.25)
err.set_ylabel('Errors [m]')
err.set_xlabel('Time [sec]')
err.legend()

# ax2 = fig.add_subplot(gs0[:8,6:12])
# pos = fig.add_subplot(gs0[8:14,:8])
# vel = fig.add_subplot(gs0[14:,:8])

# pos_err = fig.add_subplot(gs0[8:14,8:])
# vel_err = fig.add_subplot(gs0[14:,8:])
# ax5 = fig.add_subplot(gs0[:8,12:])

# gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

# ax1 = fig.add_subplot(gs00[:-1, :])
# ax2 = fig.add_subplot(gs00[-1, :-1])
# ax3 = fig.add_subplot(gs00[-1, -1])

# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(3, 3)

# ax4 = fig.add_subplot(gs01[:, :-1])
# ax5 = fig.add_subplot(gs01[:-1, -1])
# ax6 = fig.add_subplot(gs01[-1, -1])

# plt.suptitle("GridSpec Inside GridSpec")

# grasp_time = mean_df[mean_df[gripper_state] == 0].index[0]

# plot_mean_std(pos, mean_df, mean_df[mocap_drone_y], std_df[mocap_drone_y], label="Drone y")
# plot_mean_std(pos, mean_df, mean_df[mocap_drone_z], std_df[mocap_drone_z], label="Drone z")

# plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_y], std_df[:grasp_time][mocap_target_y], label="Target y")
# plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_z] + offset, std_df[:grasp_time][mocap_target_z], label="Target z")
# pos.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# pos.legend()
# pos.set_ylabel("Position [m]")
# pos.tick_params(labelbottom=False)

# plot_mean_std(vel, mean_df, mean_df[ulog_vx], std_df[ulog_vx], label="Drone vy")
# plot_mean_std(vel, mean_df, -mean_df[ulog_vz], std_df[ulog_vz], label="Drone vz")

# # plot_mean_std(ax4, mean_df, mean_df[mocap_target_vx], std_df[mocap_target_vx], label="Target vx")
# plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vy], std_df[:grasp_time][mocap_target_vy], label="Target vy")
# plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vz], std_df[:grasp_time][mocap_target_vz], label="Target vz")
# vel.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# vel.set_ylabel("Velocity [m/s]")
# vel.legend()

# plot_mean_std(pos_err, mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="Error")

# pos_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# pos_err.set_ylabel("Position Error [m]")
# pos_err.tick_params(labelbottom=False)

# plot_mean_std(vel_err, mean_df, mean_df[ulog_vel_error_norm], std_df[ulog_vel_error_norm], label="Error")
# vel_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
# vel_err.set_ylabel("Velocity Error [m/s]")


# bar.bar(["Slow", "Fast"], [10,6])
# bar.set(ylabel="Successes")


N = 3
vision = (10, 9, 6)
mocap = (10, 7, 8)
ind = np.arange(N)
width = 0.3       

# Plotting
bar.bar(ind, vision , width, label='Vision')
bar.bar(ind + width, mocap, width, label='Mocap')

# plt.xlabel('Here goes x-axis label')
bar.set_ylabel('Success')
# plt.title('Here goes title of the plot')
bar.set_xticks(ind + width / 2, ('Pepsi', 'Medkit', 'Cardboard Box'))

bar.legend(loc='best')
bar.axhline(y=10, color='0', ls='--', lw=2)


ax1.imshow(medkit_img)
ax1.grid(False)
ax1.tick_params(labelbottom=False, labelleft=False)

ax2.imshow(pepsi_img)
ax2.grid(False)
ax2.tick_params(labelbottom=False, labelleft=False)

ax3.imshow(cardboard_img)
ax3.grid(False)
ax3.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
# format_axes(fig)
plt.show()


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

fig, ax = plt.subplots(1,1, figsize=(20,10))

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

# plot_mean_std(ax[3][0], mean_df, std_df, "alignment_error_z", label="VIO Drift Error z")ulog_position_sp_x
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

# # 3D plot

# +
ulog_location = "../mocap_a1_slow"
ulog_result_location = "../log_output"
bag_location = "../mocap_a1_slow"
bag_result_location = "../mocap_a1_slow"

dfs_a1_slow = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs_a1_slow, mocap_drone_x)


ulog_location = "../mocap_a1_fast"
ulog_result_location = "../log_output"
bag_location = "../mocap_a1_fast"
bag_result_location = "../mocap_a1_fast"

dfs_a1_fast = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs_a1_fast, mocap_drone_x)


# create_aligned_mavros(dfs)

# +
sns.set_style('ticks')
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect([1,1,1])
ax.view_init(elev=20, azim=0)


xs = []
ys = []
zs = []
for df in dfs:
    df_sliced = df[-5:9]
    drone_traj = np.array([df_sliced[mocap_drone_x], df_sliced[mocap_drone_y], df_sliced[mocap_drone_z]]).T
    df_sliced_2 = df[-5:4]
    target_traj = np.array([df_sliced_2[mocap_target_x], df_sliced_2[mocap_target_y], df_sliced_2[mocap_target_z]]).T
#     print(drone_traj)
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], color='g')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r')
    xs += drone_traj[:,0].tolist()
    ys += drone_traj[:,1].tolist()
    zs += drone_traj[:,2].tolist()
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
ax.set_zlim([0, 1.4])


# -

add_velocities(dfs_a1_slow)
add_velocities(dfs_a1_fast)
mean_df_a1_slow, std_df_a1_slow = add_errors(dfs_a1_slow, vision=False)
mean_df_a1_fast, std_df_a1_fast = add_errors(dfs_a1_fast, vision=False)

# +


sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(2,1, figsize=(3,6))

# plot_mean_std(ax[0], mean_df, (mean_df[mocap_drone_x]), std_df[mocap_drone_x], label="X")
# plot_mean_std(ax[0], mean_df_a1_slow, (mean_df[mocap_target_y]), std_df[mocap_target_y], label="Y")
# plot_mean_std(ax[0], mean_df_a1_slow, (mean_df[mocap_target_z]), std_df[mocap_target_z], label="Z")

sliced_mean_df_a1_slow = mean_df_a1_slow[:4]
sliced_std_df_a1_slow = std_df_a1_slow[:4]

sliced_mean_df_a1_fast = mean_df_a1_fast[:4]
sliced_std_df_a1_fast = std_df_a1_fast[:4]

# plot_mean_std(ax[0], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_x], sliced_std_df_a1_slow[mocap_target_x], label="Slow X")
plot_mean_std(ax[0], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_y], sliced_std_df_a1_slow[mocap_target_y], label="Slow Y")


# plot_mean_std(ax[0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_x], sliced_std_df_a1_fast[mocap_target_x], label="Fast X")
plot_mean_std(ax[0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_y], sliced_std_df_a1_fast[mocap_target_y], label="Fast Y")
# plot_mean_std(ax[0], cardboard_box_mean_df, np.abs(cardboard_box_mean_df[ulog_error_z]), cardboard_box_std_df[ulog_error_z], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, np.abs(pepsi_mean_df[ulog_error_z]), pepsi_std_df[ulog_error_z], label="2 Liter Bottle")

ax[0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0].axvline(x=[4], color='0', ls='--', lw=2)
ax[0].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
ax[0].set_ylabel('Pos. [m]')
# ax[0].set_xlabel('Time [s]')
ax[0].set_xticklabels([])
ax[0].legend()


# plot_mean_std(ax[1], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_vx], sliced_std_df_a1_slow[mocap_target_vx], label="Slow X")
plot_mean_std(ax[1], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_vy], sliced_std_df_a1_slow[mocap_target_vy], label="Slow")


# plot_mean_std(ax[1], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vx], sliced_std_df_a1_fast[mocap_target_vx], label="Fast X")
plot_mean_std(ax[1], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vy], sliced_std_df_a1_fast[mocap_target_vy], label="Fast")


ax[1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
ax[1].set_ylabel('Vel. [m/s]')
ax[1].set_xlabel('Time [s]')
# ax[1].set_xticklabels([])
ax[1].legend()

# +
ulog_location = "../mocap_turntable_05ms"
ulog_result_location = "../log_output"
bag_location = "../mocap_turntable_05ms"
bag_result_location = "../mocap_turntable_05ms"

dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs, mocap_drone_x)

ulog_location = "../mocap_turntable_1ms"
ulog_result_location = "../log_output"
bag_location = "../mocap_turntable_1ms"
bag_result_location = "../mocap_turntable_1ms"

dfs_mid = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs_mid, mocap_drone_x)

ulog_location = "../mocap_turntable_15ms"
ulog_result_location = "../log_output"
bag_location = "../mocap_turntable_15ms"
bag_result_location = "../mocap_turntable_15ms"

dfs_fast = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs_fast, mocap_drone_x)

# +
add_velocities(dfs)
add_velocities(dfs_mid)
add_velocities(dfs_fast)

odds_dfs = [dfs[i] for i in range(1, len(dfs), 2)]
odds_dfs_mid = [dfs_mid[i] for i in range(1, len(dfs_mid), 2)]
odds_dfs_fast = [dfs_fast[i] for i in range(1, len(dfs_fast), 2)]

mean_df, std_df = add_errors(odds_dfs, vision=False)
mean_df_mid, std_df_mid = add_errors(odds_dfs_mid, vision=False)
mean_df_fast, std_df_fast = add_errors(odds_dfs_fast, vision=False)

# mean_df, std_df = add_errors(dfs, vision=False)
# mean_df_mid, std_df_mid = add_errors(dfs, vision=False)
# mean_df_fast, std_df_fast = add_errors(dfs, vision=False)

# +
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(1,2, figsize=(12,3))
# plot_mean_std(ax[0], medkit_mean_df, medkit_mean_df[ulog_pos_error_norm], medkit_std_df[ulog_pos_error_norm], label="Medkit")
# plot_mean_std(ax[0], cardboard_box_mean_df, cardboard_box_mean_df[ulog_pos_error_norm], cardboard_box_std_df[ulog_pos_error_norm], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, pepsi_mean_df[ulog_pos_error_norm], pepsi_std_df[ulog_pos_error_norm], label="2 Liter Bottle")
plot_mean_std(ax[0], mean_df_a1_slow, mean_df_a1_slow[ulog_pos_error_norm], std_df_a1_slow[ulog_pos_error_norm], label="Slow")
plot_mean_std(ax[0], mean_df_a1_fast, mean_df_a1_fast[ulog_pos_error_norm], std_df_a1_fast[ulog_pos_error_norm], label="Fast")
# plot_mean_std(ax[0], cardboard_box_mean_df, np.abs(cardboard_box_mean_df[ulog_error_z]), cardboard_box_std_df[ulog_error_z], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, np.abs(pepsi_mean_df[ulog_error_z]), pepsi_std_df[ulog_error_z], label="2 Liter Bottle")

ax[0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0].axvline(x=[4], color='0', ls='--', lw=2)
ax[0].set_xlim(0, 14)
ax[0].set_ylim(0, 0.4)
ax[0].set_ylabel('Pos. Errors [m]')
ax[0].set_xlabel('Time [s]')
# ax[0].set_xticklabels([])
ax[0].legend()

plot_mean_std(ax[1], mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="0.5 m/s")
plot_mean_std(ax[1], mean_df_mid, mean_df_mid[ulog_pos_error_norm], std_df_mid[ulog_pos_error_norm], label="1 m/s")
plot_mean_std(ax[1], mean_df_fast, mean_df_fast[ulog_pos_error_norm], std_df_fast[ulog_pos_error_norm], label="1.5 m/s")

ax[1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1].set_xlim(0, 14)
ax[1].set_ylim(0, 0.3)
# ax[1].set_ylabel('Pos. Errors [m]')
ax[1].set_xlabel('Time [s]')
# ax[1].set_xticklabels([])
ax[1].legend()


sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(2,2, figsize=(6,4))

# plot_mean_std(ax[0], mean_df, (mean_df[mocap_drone_x]), std_df[mocap_drone_x], label="X")
# plot_mean_std(ax[0], mean_df_a1_slow, (mean_df[mocap_target_y]), std_df[mocap_target_y], label="Y")
# plot_mean_std(ax[0], mean_df_a1_slow, (mean_df[mocap_target_z]), std_df[mocap_target_z], label="Z")



sliced_mean_df_a1_slow = mean_df_a1_slow[:4]
sliced_std_df_a1_slow = std_df_a1_slow[:4]

sliced_mean_df_a1_fast = mean_df_a1_fast[:4]
sliced_std_df_a1_fast = std_df_a1_fast[:4]

# plot_mean_std(ax[0], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_x], sliced_std_df_a1_slow[mocap_target_x], label="Slow X")
plot_mean_std(ax[0][0], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_y], sliced_std_df_a1_slow[mocap_target_y], label="Slow")


# plot_mean_std(ax[0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_x], sliced_std_df_a1_fast[mocap_target_x], label="Fast X")
plot_mean_std(ax[0][0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_y], sliced_std_df_a1_fast[mocap_target_y], label="Fast")
# plot_mean_std(ax[0], cardboard_box_mean_df, np.abs(cardboard_box_mean_df[ulog_error_z]), cardboard_box_std_df[ulog_error_z], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, np.abs(pepsi_mean_df[ulog_error_z]), pepsi_std_df[ulog_error_z], label="2 Liter Bottle")

ax[0][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[0][0].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
ax[0][0].set_ylabel('Pos. [m]')
# ax[0].set_xlabel('Time [s]')
# ax[0][0].set_xticklabels([])
ax[0][0].legend()


# plot_mean_std(ax[1], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_vx], sliced_std_df_a1_slow[mocap_target_vx], label="Slow X")
plot_mean_std(ax[1][0], sliced_mean_df_a1_slow, sliced_mean_df_a1_slow[mocap_target_vy], sliced_std_df_a1_slow[mocap_target_vy], label="Slow")


# plot_mean_std(ax[1], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vx], sliced_std_df_a1_fast[mocap_target_vx], label="Fast X")
plot_mean_std(ax[1][0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vy], sliced_std_df_a1_fast[mocap_target_vy], label="Fast")


ax[1][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[1][0].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
ax[1][0].set_ylabel('Vel. [m/s]')
ax[1][0].set_xlabel('Time [s]')
# ax[1].set_xticklabels([])
# ax[1][0].legend()

sliced_mean_df_slow = mean_df[:4]
sliced_std_df_slow = std_df[:4]

sliced_mean_df_a1_mid = mean_df_mid[:4]
sliced_std_df_a1_mid = std_df_mid[:4]

sliced_mean_df_a1_fast = mean_df_fast[:4]
sliced_std_df_a1_fast = std_df_fast[:4]

plot_mean_std(ax[0][1], sliced_mean_df_slow, sliced_mean_df_slow[mocap_target_x], sliced_std_df_slow[mocap_target_x], label="X")
plot_mean_std(ax[0][1], sliced_mean_df_slow, sliced_mean_df_slow[mocap_target_y], sliced_std_df_slow[mocap_target_y], label="Y")


# plot_mean_std(ax[0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_x], sliced_std_df_a1_fast[mocap_target_x], label="Fast X")
# plot_mean_std(ax[0], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_y], sliced_std_df_a1_fast[mocap_target_y], label="Fast Y")
# plot_mean_std(ax[0], cardboard_box_mean_df, np.abs(cardboard_box_mean_df[ulog_error_z]), cardboard_box_std_df[ulog_error_z], label="Cardboard Box")
# plot_mean_std(ax[0], pepsi_mean_df, np.abs(pepsi_mean_df[ulog_error_z]), pepsi_std_df[ulog_error_z], label="2 Liter Bottle")

ax[0][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[0][1].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
# ax[0].set_ylabel('Pos. [m]')
# ax[0].set_xlabel('Time [s]')
ax[0][1].set_xticklabels([])
ax[0][1].legend()


plot_mean_std(ax[1][1], sliced_mean_df_slow, sliced_mean_df_slow[mocap_target_vx], sliced_std_df_slow[mocap_target_vx], label="X")
plot_mean_std(ax[1][1], sliced_mean_df_slow, sliced_mean_df_slow[mocap_target_vy], sliced_std_df_slow[mocap_target_vy], label="Y")


# plot_mean_std(ax[1], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vx], sliced_std_df_a1_fast[mocap_target_vx], label="Fast X")
# plot_mean_std(ax[1], sliced_mean_df_a1_fast, sliced_mean_df_a1_fast[mocap_target_vy], sliced_std_df_a1_fast[mocap_target_vy], label="Fast")


ax[1][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1][1].set_xlim(-5, 4.5)
# ax[0].set_ylim(0, 0.25)
# ax[1].set_ylabel('Vel. [m/s]')
ax[1][1].set_xlabel('Time [s]')
# ax[1].set_xticklabels([])
# ax[1][1].legend()
plt.tight_layout()

fig, ax = plt.subplots(1,1, figsize=(5,4))
N = 5
mocap = (10, 6, 10, 9, 7)
ind = np.arange(N)
width = 0.6    

# Plotting
# ax.bar(ind, vision , width, label='Vision')
ax.bar(ind, mocap, width)
ax.set_ylabel('Success')
# plt.title('Here goes title of the plot')
ax.set_xticks(ind, ('Slow', 'Fast', '0.5m/s', '1.0m/s', '1.5m/s'))
# plt.xticks(rotation = 45)

ax.legend(loc='best')
ax.axhline(y=10, color='0', ls='--', lw=2)

# +


sns.set_style('ticks')
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect([1,1,1])
ax.view_init(elev=90, azim=-90)

odds_dfs = [dfs[i] for i in range(1, len(dfs), 2)]
odds_dfs_mid = [dfs_mid[i] for i in range(1, len(dfs_mid), 2)]
odds_dfs_fast = [dfs_fast[i] for i in range(1, len(dfs_fast), 2)]

slow_turntable_mean_df, slow_turntable_std_df = add_errors(odds_dfs, vision=False)
mid_turntable_mean_df, mid_turntable_std_df = add_errors(odds_dfs_mid, vision=False)
fast_turntable_mean_df, fast_turntable_std_df = add_errors(odds_dfs_fast, vision=False)

all_dfs = odds_dfs + odds_dfs_fast


xs = []
ys = []
zs = []
for df in odds_dfs:
    df_sliced = df[-5:14]
    drone_traj = np.array([df_sliced[mocap_drone_x], df_sliced[mocap_drone_y], df_sliced[mocap_drone_z]]).T
    df_sliced_2 = df[-5:4]
    target_traj = np.array([df_sliced_2[mocap_target_x], df_sliced_2[mocap_target_y], df_sliced_2[mocap_target_z]]).T
    #     print(drone_traj)
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], color='g')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r')
    xs += drone_traj[:,0].tolist()
    ys += drone_traj[:,1].tolist()
    zs += drone_traj[:,2].tolist()
    
    
for df in odds_dfs_mid:
    df_sliced = df[-5:14]
    drone_traj = np.array([df_sliced[mocap_drone_x], df_sliced[mocap_drone_y], df_sliced[mocap_drone_z]]).T
    df_sliced_2 = df[-5:4]
    target_traj = np.array([df_sliced_2[mocap_target_x], df_sliced_2[mocap_target_y], df_sliced_2[mocap_target_z]]).T
    #     print(drone_traj)
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], color='r')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r')

    xs += drone_traj[:,0].tolist()
    ys += drone_traj[:,1].tolist()
    zs += drone_traj[:,2].tolist()
    
    
for df in odds_dfs_fast:
    df_sliced = df[-5:14]
    drone_traj = np.array([df_sliced[mocap_drone_x], df_sliced[mocap_drone_y], df_sliced[mocap_drone_z]]).T
    df_sliced_2 = df[-5:4]
    target_traj = np.array([df_sliced_2[mocap_target_x], df_sliced_2[mocap_target_y], df_sliced_2[mocap_target_z]]).T
    #     print(drone_traj)
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], color='b')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r')
    xs += drone_traj[:,0].tolist()
    ys += drone_traj[:,1].tolist()
    zs += drone_traj[:,2].tolist()
    
    
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
ax.set_zlim([0, 1.4])


# +
##############################################################3
fig, ax = plt.subplots(5,2, figsize=(12,18))

medkit_mean_df_sliced = medkit_mean_df[0:15]
medkit_std_df_sliced = medkit_std_df[0:15]
plot_mean_std(ax[0][0], medkit_mean_df_sliced, medkit_mean_df_sliced[ulog_error_x], medkit_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[0][0], medkit_mean_df_sliced, medkit_mean_df_sliced[ulog_error_y], medkit_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[0][0], medkit_mean_df_sliced, medkit_mean_df_sliced[ulog_error_z], medkit_std_df_sliced[ulog_error_z], label="Z")

ax[0][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[0][0].set_xlim(0, 15)
# ax[0][0].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][0].set_xlabel('Time [s]')
ax[0][0].set_title("Medkit 0.5 m/s")
# ax.set_xticklabels([])
ax[0][0].legend()

cardboard_box_mean_df_sliced = cardboard_box_mean_df[0:15]
cardboard_box_std_df_sliced = cardboard_box_std_df[0:15]
plot_mean_std(ax[1][0], cardboard_box_mean_df_sliced, cardboard_box_mean_df_sliced[ulog_error_x], cardboard_box_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[1][0], cardboard_box_mean_df_sliced, cardboard_box_mean_df_sliced[ulog_error_y], cardboard_box_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[1][0], cardboard_box_mean_df_sliced, cardboard_box_mean_df_sliced[ulog_error_z], cardboard_box_std_df_sliced[ulog_error_z], label="Z")

ax[1][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[1][0].set_xlim(0, 15)
# ax[0][0].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][0].set_xlabel('Time [s]')
ax[1][0].set_title("Cardboard Box 0.5 m/s")

pepsi_mean_df_sliced = pepsi_mean_df[0:15]
pepsi_std_df_sliced = pepsi_std_df[0:15]
plot_mean_std(ax[2][0], pepsi_mean_df_sliced, pepsi_mean_df_sliced[ulog_error_x], pepsi_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[2][0], pepsi_mean_df_sliced, pepsi_mean_df_sliced[ulog_error_y], pepsi_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[2][0], pepsi_mean_df_sliced, pepsi_mean_df_sliced[ulog_error_z], pepsi_std_df_sliced[ulog_error_z], label="Z")

ax[2][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[2][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[2][0].set_xlim(0, 15)
# ax[0][0].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][0].set_xlabel('Time [s]')
ax[2][0].set_title("2 Liter Bottle 0.5 m/s")

pepsi_mid_mean_df_sliced = pepsi_mid_mean_df[0:15]
pepsi_mid_std_df_sliced = pepsi_mid_std_df[0:15]
plot_mean_std(ax[3][0], pepsi_mid_mean_df_sliced, pepsi_mid_mean_df_sliced[ulog_error_x], pepsi_mid_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[3][0], pepsi_mid_mean_df_sliced, pepsi_mid_mean_df_sliced[ulog_error_y], pepsi_mid_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[3][0], pepsi_mid_mean_df_sliced, pepsi_mid_mean_df_sliced[ulog_error_z], pepsi_mid_std_df_sliced[ulog_error_z], label="Z")

ax[3][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[3][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[3][0].set_xlim(0, 15)
# ax[0][0].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][0].set_xlabel('Time [s]')
ax[3][0].set_title("2 Liter Bottle 1.25 m/s")

pepsi_fast_mean_df_sliced = pepsi_fast_mean_df[0:15]
pepsi_fast_std_df_sliced = pepsi_fast_std_df[0:15]
plot_mean_std(ax[4][0], pepsi_fast_mean_df_sliced, pepsi_fast_mean_df_sliced[ulog_error_x], pepsi_fast_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[4][0], pepsi_fast_mean_df_sliced, pepsi_fast_mean_df_sliced[ulog_error_y], pepsi_fast_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[4][0], pepsi_fast_mean_df_sliced, pepsi_fast_mean_df_sliced[ulog_error_z], pepsi_fast_std_df_sliced[ulog_error_z], label="Z")

ax[4][0].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[4][0].axvline(x=[4], color='0', ls='--', lw=2)
ax[4][0].set_xlim(0, 15)
# ax[0][0].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
ax[4][0].set_xlabel('Time [s]')
ax[4][0].set_title("2 Liter Bottle 2 m/s")


a1_slow_mean_df_sliced = mean_df_a1_slow[0:15]
a1_slow_std_df_sliced = std_df_a1_slow[0:15]
plot_mean_std(ax[0][1], a1_slow_mean_df_sliced, -a1_slow_mean_df_sliced[ulog_error_y], a1_slow_std_df_sliced[ulog_error_y], label="X")
plot_mean_std(ax[0][1], a1_slow_mean_df_sliced, a1_slow_mean_df_sliced[ulog_error_x], a1_slow_std_df_sliced[ulog_error_x], label="Y")
plot_mean_std(ax[0][1], a1_slow_mean_df_sliced, a1_slow_mean_df_sliced[ulog_error_z], a1_slow_std_df_sliced[ulog_error_z], label="Z")

ax[0][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[0][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[0][1].set_xlim(0, 15)
ax[0][1].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][1].set_xlabel('Time [s]')
ax[0][1].set_title("A1 Slow")

a1_fast_mean_df_sliced = mean_df_a1_fast[0:15]
a1_fast_std_df_sliced = std_df_a1_fast[0:15]
plot_mean_std(ax[1][1], a1_fast_mean_df_sliced, -a1_fast_mean_df_sliced[ulog_error_y], a1_fast_std_df_sliced[ulog_error_y], label="X")
plot_mean_std(ax[1][1], a1_fast_mean_df_sliced, a1_fast_mean_df_sliced[ulog_error_x], a1_fast_std_df_sliced[ulog_error_x], label="Y")
plot_mean_std(ax[1][1], a1_fast_mean_df_sliced, a1_fast_mean_df_sliced[ulog_error_z], a1_fast_std_df_sliced[ulog_error_z], label="Z")

ax[1][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[1][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[1][1].set_xlim(0, 15)
# ax[1][1].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][1].set_xlabel('Time [s]')
ax[1][1].set_title("A1 Fast")


slow_turntable_mean_df_sliced = slow_turntable_mean_df[0:15]
slow_turntable_std_df_sliced = slow_turntable_std_df[0:15]
plot_mean_std(ax[2][1], slow_turntable_mean_df_sliced, slow_turntable_mean_df_sliced[ulog_error_x], slow_turntable_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[2][1], slow_turntable_mean_df_sliced, slow_turntable_mean_df_sliced[ulog_error_y], slow_turntable_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[2][1], slow_turntable_mean_df_sliced, slow_turntable_mean_df_sliced[ulog_error_z], slow_turntable_std_df_sliced[ulog_error_z], label="Z")

ax[2][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[2][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[2][1].set_xlim(0, 15)
# ax[1][1].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][1].set_xlabel('Time [s]')
ax[2][1].set_title("Turntable 0.5 m/s")


mid_turntable_mean_df_sliced = mid_turntable_mean_df[0:15]
mid_turntable_std_df_sliced = mid_turntable_std_df[0:15]
plot_mean_std(ax[3][1], mid_turntable_mean_df_sliced, mid_turntable_mean_df_sliced[ulog_error_x], mid_turntable_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[3][1], mid_turntable_mean_df_sliced, mid_turntable_mean_df_sliced[ulog_error_y], mid_turntable_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[3][1], mid_turntable_mean_df_sliced, mid_turntable_mean_df_sliced[ulog_error_z], mid_turntable_std_df_sliced[ulog_error_z], label="Z")

ax[3][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[3][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[3][1].set_xlim(0, 15)
# ax[1][1].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
# ax[0][1].set_xlabel('Time [s]')
ax[3][1].set_title("Turntable 1.0 m/s")

fast_turntable_mean_df_sliced = fast_turntable_mean_df[0:15]
fast_turntable_std_df_sliced = fast_turntable_std_df[0:15]
plot_mean_std(ax[4][1], fast_turntable_mean_df_sliced, fast_turntable_mean_df_sliced[ulog_error_x], fast_turntable_std_df_sliced[ulog_error_x], label="X")
plot_mean_std(ax[4][1], fast_turntable_mean_df_sliced, fast_turntable_mean_df_sliced[ulog_error_y], fast_turntable_std_df_sliced[ulog_error_y], label="Y")
plot_mean_std(ax[4][1], fast_turntable_mean_df_sliced, fast_turntable_mean_df_sliced[ulog_error_z], fast_turntable_std_df_sliced[ulog_error_z], label="Z")

ax[4][1].axvline(x=[0], color='0.5', ls='--', lw=2)
ax[4][1].axvline(x=[4], color='0', ls='--', lw=2)
ax[4][1].set_xlim(0, 15)
# ax[1][1].set_ylim(-0.25, 0.25)
# autoscale_y(ax[0][0])
# ax[0][0].set_ylabel('Pos. Errors [m]')
ax[4][1].set_xlabel('Time [s]')
ax[4][1].set_title("Turntable 1.5 m/s")


fig.tight_layout()



# -

# # Vision Speed Composite

ulog_location = "../vision_pepsi_2ms"
ulog_result_location = "../log_output"
bag_location = "../vision_pepsi_2ms"
bag_result_location = "../vision_pepsi_2ms"
dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_in_x)
validate_alignment(dfs, mavros_in_x)

# +
for df in dfs:
#     df = add_error_columns(df, [mavros_in_aligned_x, mavros_in_aligned_y, mavros_in_aligned_z],
#                   [mocap_drone_x, mocap_drone_y, mocap_drone_z],
#                   [alignment_error_x, alignment_error_y, alignment_error_z],
#                    alignment_error_norm
#                   )
    
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
    df = add_error_columns(df, [ulog_x, ulog_y, ulog_z],
                  [ulog_sp_x, ulog_sp_y, ulog_sp_z],
                  [ulog_error_x, ulog_error_y, ulog_error_z],
                   ulog_pos_error_norm
                  )
    
    df = add_error_columns(df, [ulog_vx, ulog_vy, ulog_vz],
                  [ulog_sp_vx, ulog_sp_vy, ulog_sp_vz],
                  [ulog_error_vx, ulog_error_vy, ulog_error_vz],
                   ulog_vel_error_norm
                  )
        
mean_df, std_df = get_aggregate_df(dfs)
mean_df = mean_df[-5:8]
std_df = std_df[-5:8]

# +
slow_img = plt.imread("../vision_pepsi_2ms/vision_pepsi_2ms.png")
slow_img = slow_img[:1600,:]
offset = 0.2
sns.set(font_scale=1.2)
def format_axes(fig):
    labels = ["A","B", "C", "D", "E", "F", "G"]
    for i, ax in enumerate(fig.axes):
        ax.text(1.0, 1.0, labels[i], va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
fig = plt.figure(figsize=(20, 16))

gs0 = gridspec.GridSpec(20, 16, figure=fig)
ax1 = fig.add_subplot(gs0[:8,:12])
# ax2 = fig.add_subplot(gs0[:8,6:12])
pos = fig.add_subplot(gs0[8:14,:8])
vel = fig.add_subplot(gs0[14:,:8])

pos_err = fig.add_subplot(gs0[8:14,8:])
vel_err = fig.add_subplot(gs0[14:,8:])
bar = fig.add_subplot(gs0[:8,12:])

# gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

# ax1 = fig.add_subplot(gs00[:-1, :])
# ax2 = fig.add_subplot(gs00[-1, :-1])
# ax3 = fig.add_subplot(gs00[-1, -1])

# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(3, 3)

# ax4 = fig.add_subplot(gs01[:, :-1])
# ax5 = fig.add_subplot(gs01[:-1, -1])
# ax6 = fig.add_subplot(gs01[-1, -1])

# plt.suptitle("GridSpec Inside GridSpec")

grasp_time = mean_df[mean_df[gripper_state] == 0].index[0]

plot_mean_std(pos, mean_df, mean_df[mocap_drone_x], std_df[mocap_drone_x], label="Drone x")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_y], std_df[mocap_drone_y], label="Drone y")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_z], std_df[mocap_drone_z], label="Drone z")

# plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_x], std_df[:grasp_time][mocap_target_x], label="Target x")
# plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_y], std_df[:grasp_time][mocap_target_y], label="Target y")
# plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_z] + offset, std_df[:grasp_time][mocap_target_z], label="Target z")
pos.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos.legend()
pos.set_ylabel("Position [m]")
pos.tick_params(labelbottom=False)

# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vx], std_df[mocap_drone_x], label="Drone vx")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vy], std_df[mocap_drone_vy], label="Drone vy")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vz], std_df[mocap_drone_vz], label="Drone vz")
plot_mean_std(vel, mean_df, mean_df[ulog_vy], std_df[ulog_vy], label="Drone vx")
plot_mean_std(vel, mean_df, mean_df[ulog_vx], std_df[ulog_vx], label="Drone vy")
plot_mean_std(vel, mean_df, -mean_df[ulog_vz], std_df[ulog_vz], label="Drone vz")



# plot_mean_std(ax4, mean_df, mean_df[mocap_target_vx], std_df[mocap_target_vx], label="Target vx")
# plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vy], std_df[:grasp_time][mocap_target_vy], label="Target vy")
# plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vz], std_df[:grasp_time][mocap_target_vz], label="Target vz")
vel.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel.set_ylabel("Velocity [m/s]")
vel.legend()
# ax3.tick_params(labelbottom=False)

plot_mean_std(pos_err, mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="Error")

pos_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos_err.set_ylabel("Position Error [m]")
# pos_err.yaxis.set_label_position("right")
# pos_err.yaxis.tick_right()

pos_err.tick_params(labelbottom=False)

plot_mean_std(vel_err, mean_df, mean_df[ulog_vel_error_norm], std_df[ulog_vel_error_norm], label="Error")
vel_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel_err.set_ylabel("Velocity Error [m/s]")
# vel_err.yaxis.set_label_position("right")
# vel_err.yaxis.tick_right()
# vel_err.tick_params(labelbottom=False)


N = 3
vision = (10, 10, 8)
mocap = (10, 9, 6)
ind = np.arange(N)
width = 0.3       

# Plotting
bar.bar(ind, vision , width, label='Vision')
bar.bar(ind + width, mocap, width, label='Mocap')

# plt.xlabel('Here goes x-axis label')
plt.ylabel('Success')
# plt.title('Here goes title of the plot')
plt.xticks(ind + width / 2, ('0.5 m/s', '1.25 m/s', '2 m/s'))

plt.legend(loc='best')
bar.axhline(y=10, color='0', ls='--', lw=2)

ax1.imshow(slow_img)
ax1.grid(False)
ax1.tick_params(labelbottom=False, labelleft=False)
ax2.imshow(slow_img)
ax2.grid(False)
ax2.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
# format_axes(fig)
plt.show()
# -

# # Mocap Turntable Composite

ulog_location = "../mocap_turntable_15ms"
ulog_result_location = "../log_output"
bag_location = "../mocap_turntable_15ms"
bag_result_location = "../mocap_turntable_15ms"
dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs, mocap_drone_x)

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
for df in dfs:
    df = add_error_columns(df, [ulog_x, ulog_y, ulog_z],
                  [ulog_sp_x, ulog_sp_y, ulog_sp_z],
                  [ulog_error_x, ulog_error_y, ulog_error_z],
                   ulog_pos_error_norm
                  )
    
    df = add_error_columns(df, [ulog_vx, ulog_vy, ulog_vz],
                  [ulog_sp_vx, ulog_sp_vy, ulog_sp_vz],
                  [ulog_error_vx, ulog_error_vy, ulog_error_vz],
                   ulog_vel_error_norm
                  )

add_velocities(dfs)
evens_dfs = [dfs[i] for i in range(0, len(dfs), 2)]
odds_df = [dfs[i] for i in range(1, len(dfs), 2)]
mean_df, std_df = get_aggregate_df(evens_dfs)
mean_df = mean_df[-5:8]
std_df = std_df[-5:8]

# +
slow_img = plt.imread("../mocap_turntable_15ms/15ms.png")
slow_img = slow_img[:1600,:]
offset = 0.2
sns.set(font_scale=1.2)
def format_axes(fig):
    labels = ["A","B", "C", "D", "E", "F", "G"]
    for i, ax in enumerate(fig.axes):
        ax.text(1.0, 1.0, labels[i], va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
fig = plt.figure(figsize=(20, 16))

gs0 = gridspec.GridSpec(20, 16, figure=fig)
ax1 = fig.add_subplot(gs0[:8,:12])
# ax2 = fig.add_subplot(gs0[:8,6:12])
pos = fig.add_subplot(gs0[8:14,:8])
vel = fig.add_subplot(gs0[14:,:8])

pos_err = fig.add_subplot(gs0[8:14,8:])
vel_err = fig.add_subplot(gs0[14:,8:])
ax5 = fig.add_subplot(gs0[:8,12:])

# gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

# ax1 = fig.add_subplot(gs00[:-1, :])
# ax2 = fig.add_subplot(gs00[-1, :-1])
# ax3 = fig.add_subplot(gs00[-1, -1])

# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(3, 3)

# ax4 = fig.add_subplot(gs01[:, :-1])
# ax5 = fig.add_subplot(gs01[:-1, -1])
# ax6 = fig.add_subplot(gs01[-1, -1])

# plt.suptitle("GridSpec Inside GridSpec")

grasp_time = mean_df[mean_df[gripper_state] == 0].index[0]

plot_mean_std(pos, mean_df, mean_df[mocap_drone_x], std_df[mocap_drone_x], label="Drone x")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_y], std_df[mocap_drone_y], label="Drone y")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_z], std_df[mocap_drone_z], label="Drone z")

plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_x], std_df[:grasp_time][mocap_target_x], label="Target x")
plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_y], std_df[:grasp_time][mocap_target_y], label="Target y")
plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_z] + offset, std_df[:grasp_time][mocap_target_z], label="Target z")
pos.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos.legend()
pos.set_ylabel("Position [m]")
pos.tick_params(labelbottom=False)

# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vx], std_df[mocap_drone_x], label="Drone vx")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vy], std_df[mocap_drone_vy], label="Drone vy")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vz], std_df[mocap_drone_vz], label="Drone vz")
plot_mean_std(vel, mean_df, mean_df[ulog_vx], std_df[ulog_vx], label="Drone vy")
plot_mean_std(vel, mean_df, -mean_df[ulog_vz], std_df[ulog_vz], label="Drone vz")
plot_mean_std(vel, mean_df, mean_df[ulog_vy], std_df[ulog_vy], label="Drone vx")

plot_mean_std(ax4, mean_df, mean_df[mocap_target_vx], std_df[mocap_target_vx], label="Target vx")
plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vy], std_df[:grasp_time][mocap_target_vy], label="Target vy")
plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vz], std_df[:grasp_time][mocap_target_vz], label="Target vz")
vel.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel.set_ylabel("Velocity [m/s]")
vel.legend()
# ax3.tick_params(labelbottom=False)

plot_mean_std(pos_err, mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="Error")

pos_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos_err.set_ylabel("Position Error [m]")
# pos_err.yaxis.set_label_position("right")
# pos_err.yaxis.tick_right()

pos_err.tick_params(labelbottom=False)

plot_mean_std(vel_err, mean_df, mean_df[ulog_vel_error_norm], std_df[ulog_vel_error_norm], label="Error")
vel_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel_err.set_ylabel("Velocity Error [m/s]")
# vel_err.yaxis.set_label_position("right")
# vel_err.yaxis.tick_right()
# vel_err.tick_params(labelbottom=False)


ax5.bar(["0.5 m/s", "1 m/s", "1.5 m/s"], [10, 9, 7])
ax5.set(ylabel="Successes")
ax5.axhline(y=10, color='0', ls='--', lw=2)

ax1.imshow(slow_img)
ax1.grid(False)
ax1.tick_params(labelbottom=False, labelleft=False)
ax2.imshow(slow_img)
ax2.grid(False)
ax2.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
# format_axes(fig)
plt.show()

# +
ulog_location = "../mocap_a1_slow"
ulog_result_location = "../mocap_a1_slow"
bag_location = "../mocap_a1_slow"
bag_result_location = "../mocap_a1_slow"
rosbags = load_rosbags(bag_result_location, bag_location)position_topics = ["sparksdrone-world-pose.position.x",
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

# # Mocap A1 Composite

ulog_location = "../mocap_a1_fast"
ulog_result_location = "../log_output"
bag_location = "../mocap_a1_fast"
bag_result_location = "../mocap_a1_fast"
dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x)
validate_alignment(dfs, mocap_drone_x)

# +
for df in dfs:
    df = add_error_columns(df, [ulog_x, ulog_y, ulog_z],
                  [ulog_sp_x, ulog_sp_y, ulog_sp_z],
                  [ulog_error_x, ulog_error_y, ulog_error_z],
                   ulog_pos_error_norm
                  )
    
    df = add_error_columns(df, [ulog_vx, ulog_vy, ulog_vz],
                  [ulog_sp_vx, ulog_sp_vy, ulog_sp_vz],
                  [ulog_error_vx, ulog_error_vy, ulog_error_vz],
                   ulog_vel_error_norm
                  )

add_velocities(dfs)
mean_df, std_df = get_aggregate_df(dfs)
mean_df = mean_df[-3:8]
std_df = std_df[-3:8]
# -

for col in dfs[0].columns:
    print(col)

# +
slow_img = plt.imread("../mocap_a1_slow/timelapse.png")
slow_img = slow_img[:1600,:]
offset = 0.2
sns.set(font_scale=1.2)
def format_axes(fig):
    labels = ["A","B", "C", "D", "E", "F", "G"]
    for i, ax in enumerate(fig.axes):
        ax.text(1.0, 1.0, labels[i], va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)


# gridspec inside gridspec
fig = plt.figure(figsize=(20, 16))

gs0 = gridspec.GridSpec(20, 16, figure=fig)
ax1 = fig.add_subplot(gs0[:8,:12])
# ax2 = fig.add_subplot(gs0[:8,6:12])
pos = fig.add_subplot(gs0[8:14,:8])
vel = fig.add_subplot(gs0[14:,:8])

pos_err = fig.add_subplot(gs0[8:14,8:])
vel_err = fig.add_subplot(gs0[14:,8:])
ax5 = fig.add_subplot(gs0[:8,12:])

# gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

# ax1 = fig.add_subplot(gs00[:-1, :])
# ax2 = fig.add_subplot(gs00[-1, :-1])
# ax3 = fig.add_subplot(gs00[-1, -1])

# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(3, 3)

# ax4 = fig.add_subplot(gs01[:, :-1])
# ax5 = fig.add_subplot(gs01[:-1, -1])
# ax6 = fig.add_subplot(gs01[-1, -1])

# plt.suptitle("GridSpec Inside GridSpec")

grasp_time = mean_df[mean_df[gripper_state] == 0].index[0]

# plot_mean_std(ax3, mean_df, mean_df[mocap_drone_x], std_df[mocap_drone_x], label="Drone x")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_y], std_df[mocap_drone_y], label="Drone y")
plot_mean_std(pos, mean_df, mean_df[mocap_drone_z], std_df[mocap_drone_z], label="Drone z")

# plot_mean_std(ax3, mean_df, mean_df[mocap_target_x], std_df[mocap_target_x], label="Target x")
plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_y], std_df[:grasp_time][mocap_target_y], label="Target y")
plot_mean_std(pos, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_z] + offset, std_df[:grasp_time][mocap_target_z], label="Target z")
pos.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos.legend()
pos.set_ylabel("Position [m]")
pos.tick_params(labelbottom=False)

# plot_mean_std(ax4, mean_df, mean_df[mocap_drone_vx], std_df[mocap_drone_x], label="Drone vx")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vy], std_df[mocap_drone_vy], label="Drone vy")
# plot_mean_std(vel, mean_df, mean_df[mocap_drone_vz], std_df[mocap_drone_vz], label="Drone vz")
plot_mean_std(vel, mean_df, mean_df[ulog_vx], std_df[ulog_vx], label="Drone vy")
plot_mean_std(vel, mean_df, -mean_df[ulog_vz], std_df[ulog_vz], label="Drone vz")

# plot_mean_std(ax4, mean_df, mean_df[mocap_target_vx], std_df[mocap_target_vx], label="Target vx")
plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vy], std_df[:grasp_time][mocap_target_vy], label="Target vy")
plot_mean_std(vel, mean_df[:grasp_time], mean_df[:grasp_time][mocap_target_vz], std_df[:grasp_time][mocap_target_vz], label="Target vz")
vel.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel.set_ylabel("Velocity [m/s]")
vel.legend()
# ax3.tick_params(labelbottom=False)

plot_mean_std(pos_err, mean_df, mean_df[ulog_pos_error_norm], std_df[ulog_pos_error_norm], label="Error")

pos_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
pos_err.set_ylabel("Position Error [m]")
# pos_err.yaxis.set_label_position("right")
# pos_err.yaxis.tick_right()

pos_err.tick_params(labelbottom=False)

plot_mean_std(vel_err, mean_df, mean_df[ulog_vel_error_norm], std_df[ulog_vel_error_norm], label="Error")
vel_err.axvline(x=[mean_df[mean_df[gripper_state] == 0].index[0]], color='0', ls='--', lw=2)
vel_err.set_ylabel("Velocity Error [m/s]")
# vel_err.yaxis.set_label_position("right")
# vel_err.yaxis.tick_right()
# vel_err.tick_params(labelbottom=False)


ax5.bar(["Slow", "Fast"], [10,6])
ax5.set(ylabel="Successes")
ax5.axhline(y=10, color='0', ls='--', lw=2)

ax1.imshow(slow_img)
ax1.grid(False)
ax1.tick_params(labelbottom=False, labelleft=False)
ax2.imshow(slow_img)
ax2.grid(False)
ax2.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
# format_axes(fig)
plt.show()
# -

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


