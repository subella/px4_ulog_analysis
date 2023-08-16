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
from IPython.display import display
# -

sns.set(font_scale=1.5)
sns.set_style("ticks",{'axes.grid' : True})

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

mocap_aligned_x = "mocap_aligned_x"
mocap_aligned_y = "mocap_aligned_y"
mocap_aligned_z = "mocap_aligned_z"
mocap_aligned_qw = "mocap_aligned_qw"
mocap_aligned_qx = "mocap_aligned_qx"
mocap_aligned_qy = "mocap_aligned_qy"
mocap_aligned_qz = "mocap_aligned_qz"

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
mavros_in_vx = "mavros-odometry-in-twist.twist.linear.x"
mavros_in_vy = "mavros-odometry-in-twist.twist.linear.y"
mavros_in_vz = "mavros-odometry-in-twist.twist.linear.z"
mavros_in_qw = "mavros-odometry-in-pose.pose.orientation.w"
mavros_in_qx = "mavros-odometry-in-pose.pose.orientation.x"
mavros_in_qy = "mavros-odometry-in-pose.pose.orientation.y"
mavros_in_qz = "mavros-odometry-in-pose.pose.orientation.z"

mavros_out_x = "mavros-odometry-out-pose.pose.position.x"
mavros_out_y = "mavros-odometry-out-pose.pose.position.y"
mavros_out_z = "mavros-odometry-out-pose.pose.position.z"
mavros_out_vx = "mavros-odometry-out-twist.twist.linear.x"
mavros_out_vy = "mavros-odometry-out-twist.twist.linear.y"
mavros_out_vz = "mavros-odometry-out-twist.twist.linear.z"
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

mavros_wrt_tar_x = "mavros_wrt_tar_x"
mavros_wrt_tar_y = "mavros_wrt_tar_y"
mavros_wrt_tar_z = "mavros_wrt_tar_z"

mavros_wrt_tar_vx = "mavros_wrt_tar_vx"
mavros_wrt_tar_vy = "mavros_wrt_tar_vy"
mavros_wrt_tar_vz = "mavros_wrt_tar_vz"

mavros_error_x = "mavros_error_x"
mavros_error_y = "mavros_error_y"
mavros_error_z = "mavros_error_z"
mavros_error_vx = "mavros_error_vx"
mavros_error_vy = "mavros_error_vy"
mavros_error_vz = "mavros_error_vz"
mavros_pos_error_norm = "mavros_pos_error_norm"
mavros_vel_error_norm = "mavros_vel_error_norm"

mavros_wrt_tar_error_x = "mavros_wrt_tar_error_x"
mavros_wrt_tar_error_y = "mavros_wrt_tar_error_y"
mavros_wrt_tar_error_z = "mavros_wrt_tar_error_z"
mavros_wrt_tar_error_vx = "mavros_wrt_tar_error_vx"
mavros_wrt_tar_error_vy = "mavros_wrt_tar_error_vy"
mavros_wrt_tar_error_vz = "mavros_wrt_tar_error_vz"
mavros_wrt_tar_pos_error_norm = "mavros_wrt_tar_pos_error_norm"
mavros_wrt_tar_vel_error_norm = "mavros_wrt_tar_vel_error_norm"


alignment_error_x = "alignment_error_x"
alignment_error_y = "alignment_error_y"
alignment_error_z = "alignment_error_z"
alignment_error_norm = "alignment_error_norm"

mocap_alignment_error_x = "mocap_alignment_error_x"
mocap_alignment_error_y = "mocap_alignment_error_y"
mocap_alignment_error_z = "mocap_alignment_error_z"
mocap_alignment_error_norm = "mocap_alignment_error_norm"


mavros_gtsam_target_x = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.x"
mavros_gtsam_target_y = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.y"
mavros_gtsam_target_z = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.position.z"
mavros_gtsam_target_vx = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.x"
mavros_gtsam_target_vy = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.y"
mavros_gtsam_target_vz = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.linear.x"

mavros_gtsam_target_qx = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.x"
mavros_gtsam_target_qy = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.y"
mavros_gtsam_target_qz = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.z"
mavros_gtsam_target_qw = "gtsam_tracker_node-target_global_odom_estimate-pose.pose.orientation.w"
mavros_gtsam_target_omega_x = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.angular.x"
mavros_gtsam_target_omega_y = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.angular.y"
mavros_gtsam_target_omega_z = "gtsam_tracker_node-target_global_odom_estimate-twist.twist.angular.x"

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

ulog_sp_wrt_tar_x = "ulog_sp_wrt_tar_x"
ulog_sp_wrt_tar_y = "ulog_sp_wrt_tar_y"
ulog_sp_wrt_tar_z = "ulog_sp_wrt_tar_z"
ulog_sp_wrt_tar_vx = "ulog_sp_wrt_tar_vx"
ulog_sp_wrt_tar_vy = "ulog_sp_wrt_tar_vy"
ulog_sp_wrt_tar_vz = "ulog_sp_wrt_tar_vz"

ulog_x = "ulog_position_x"
ulog_y = "ulog_position_y"
ulog_z = "ulog_position_z"
ulog_vx = "ulog_position_vx"
ulog_vy = "ulog_position_vy"
ulog_vz = "ulog_position_vz"

ulog_wrt_tar_x = "ulog_wrt_tar_x"
ulog_wrt_tar_y = "ulog_wrt_tar_y"
ulog_wrt_tar_z = "ulog_wrt_tar_z"
ulog_wrt_tar_vx = "ulog_wrt_tar_vx"
ulog_wrt_tar_vy = "ulog_wrt_tar_vy"
ulog_wrt_tar_vz = "ulog_wrt_tar_vz"

ulog_error_x = "ulog_error_x"
ulog_error_y = "ulog_error_y"
ulog_error_z = "ulog_error_z"
ulog_error_vx = "ulog_error_vx"
ulog_error_vy = "ulog_error_vy"
ulog_error_vz = "ulog_error_vz"
ulog_pos_error_norm = "ulog_pos_error_norm"
ulog_vel_error_norm = "ulog_vel_error_norm"

ulog_wrt_tar_error_x = "ulog_wrt_tar_error_x"
ulog_wrt_tar_error_y = "ulog_wrt_tar_error_y"
ulog_wrt_tar_error_z = "ulog_wrt_tar_error_z"
ulog_wrt_tar_error_vx = "ulog_wrt_tar_error_vx"
ulog_wrt_tar_error_vy = "ulog_wrt_tar_error_vy"
ulog_wrt_tar_error_vz = "ulog_wrt_tar_error_vz"
ulog_wrt_tar_pos_error_norm = "ulog_wrt_tar_pos_error_norm"
ulog_wrt_tar_vel_error_norm = "ulog_wrt_tar_vel_error_norm"

grasp_segment = "grasp_state_machine_node-grasp_started-data"
gripper_state = "cmd_gripper_sub-data"

mavros_drone_sp_x = "mavros-setpoint_raw-local-position.x"
mavros_drone_sp_y = "mavros-setpoint_raw-local-position.y"
mavros_drone_sp_z = "mavros-setpoint_raw-local-position.z"

mavros_drone_sp_vx = "mavros-setpoint_raw-local-velocity.x"
mavros_drone_sp_vy = "mavros-setpoint_raw-local-velocity.y"
mavros_drone_sp_vz = "mavros-setpoint_raw-local-velocity.z"

mavros_sp_wrt_tar_x = "mavros_sp_wrt_tar_x"
mavros_sp_wrt_tar_y = "mavros_sp_wrt_tar_y"
mavros_sp_wrt_tar_z = "mavros_sp_wrt_tar_z"

mavros_sp_wrt_tar_vx = "mavros_sp_wrt_tar_vx"
mavros_sp_wrt_tar_vy = "mavros_sp_wrt_tar_vy"
mavros_sp_wrt_tar_vz = "mavros_sp_wrt_tar_vz"


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
    return rosbag
    
def format_rosbags_index(rosbags, sample_rate=10):
    for i, rosbag in enumerate(rosbags):
        rosbags[i] = format_rosbag_index(rosbag, sample_rate)
        
def reindex_rosbags(rosbags, sample_rate=10):
    for i in range(len(rosbags)):
        rosbags[i] = reindex_df(rosbags[i], 
                                int_cols=[grasp_segment, gripper_state])
 
# def get_rosbag_grasp_time(rosbag):
#     print(rosbag["cmd_gripper_sub-data"])
       
def get_rosbag_grasp_start_time(rosbag):
#     return rosbag["grasp_state_machine_node-grasp_started-data"].dropna().index[0]
    return rosbag[rosbag["cmd_gripper_sub-data"] == 0].index[0] - 4

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
    df.index = np.round(df.index, 2)
    return df

def merge_ulogs_rosbags(ulogs, rosbags):
    dfs = []
    for (ulog, rosbag) in zip(ulogs, rosbags):
        dfs.append(merge_ulog_rosbag(ulog, rosbag))
    return dfs
    
def create_dfs(ulog_location, ulog_result_location, 
              bag_location, bag_result_location,
              sample_rate=0.1,
              alignment_topic=mocap_drone_x,
              should_flip=False):
    
    rosbags = load_rosbags(bag_result_location, bag_location)
    format_rosbags(rosbags, sample_rate=10)
    if ulog_location is None:
        # Strange bug, I don't know why this is needed here.
        for rosbag in rosbags:
            rosbag.index = pd.to_timedelta(rosbag.index, unit='s')
            rosbag.index = rosbag.index.total_seconds()
            rosbag.index = np.round(rosbag.index, 2)
        return rosbags
    
    ulogs = load_ulogs(ulog_result_location, ulog_location)
    format_ulogs(ulogs)
    
    rosbags = refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic=alignment_topic, should_flip=should_flip)    
    dfs = merge_ulogs_rosbags(ulogs, rosbags)
    
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

def align_mocap_to_mavros(df, start_time=0, end_time=1):
    # Start alignment at grasp start before grasp start
    df_sliced = df.loc[start_time:]
    times = np.array(df_sliced.index)
    mavros_positions = np.array([df_sliced[mavros_wrt_tar_x], 
                                 df_sliced[mavros_wrt_tar_y],
                                 df_sliced[mavros_wrt_tar_z]]).T
    # Don't think the rotation matters for this
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

    T, (mavros_aligned, mocap_aligned) = get_aligned_trajectories(mocap_evo, mavros_evo, 
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

def create_aligned_mocap(dfs):
    for i, df in enumerate(dfs):
        start_time = -5
        end_time = 4
        T, mocap_aligned, mavros_aligned = align_mocap_to_mavros(df, start_time=start_time, end_time=end_time)
        aligned_df = pd.DataFrame()
        aligned_df.index = df[start_time:].index
        aligned_df[mocap_aligned_x] = mocap_aligned.positions_xyz[:,0]
        aligned_df[mocap_aligned_y] = mocap_aligned.positions_xyz[:,1]
        aligned_df[mocap_aligned_z] = mocap_aligned.positions_xyz[:,2]
        aligned_df[mocap_aligned_qw] = mocap_aligned.orientations_quat_wxyz[:,0]
        aligned_df[mocap_aligned_qx] = mocap_aligned.orientations_quat_wxyz[:,1]
        aligned_df[mocap_aligned_qy] = mocap_aligned.orientations_quat_wxyz[:,2]
        aligned_df[mocap_aligned_qz] = mocap_aligned.orientations_quat_wxyz[:,3]
        dfs[i] = df.join(aligned_df)
        
def refine_temporal_alignment_single(ulog, rosbag, rosbag_vo_x_topic, should_flip=False):
    # Bag always lags behind ulog, so shift bag left until error is minimized
#     rosbag.index += 1
#     rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
    step = 0.02
    shift = 0
    size = 500
    last_error = np.inf
    error = -np.inf
    while error < last_error:
        print("error", error)
        ulog_arr = -ulog["visual_odometry"]["y"].values
        if should_flip:
            ulog_arr = -ulog_arr
        bag_arr = rosbag[rosbag_vo_x_topic].values
        
        bag_center = np.where(rosbag.index == 0)[0][0]
        ulog_center = np.where(ulog["visual_odometry"].index == 0)[0][0]
        
        min_length = min(len(ulog_arr), len(bag_arr))
        if error != -np.inf:
            last_error = error
        error = np.sum(np.abs(ulog_arr[ulog_center - size: ulog_center + size] - bag_arr[bag_center - size: bag_center + size]))
        rosbag.index -= step
        rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
        
#         fig, ax = plt.subplots(1,1, figsize=(12,6))
# #         rosbag.plot(y=[rosbag_vo_x_topic], ax=ax)
# #         ulog["visual_odometry"].plot(y=["z"], ax=ax)
#         ax.plot(rosbag.index, rosbag[rosbag_vo_x_topic].values, label="Bag")
#         ax.plot(ulog["visual_odometry"].index, -ulog["visual_odometry"]["z"].values, label="Ulog")
        
    # Add back step bc algo steps one too far    
    rosbag.index += step
    rosbag = reindex_df(rosbag, int_cols=[grasp_segment, gripper_state])
    return rosbag

def refine_temporal_alignment(ulogs, rosbags, rosbag_vo_x_topic, should_flip=False): 
    for i in range(len(rosbags)):
        rosbags[i] = refine_temporal_alignment_single(ulogs[i], rosbags[i], rosbag_vo_x_topic, should_flip=should_flip)
    return rosbags
        
def validate_alignment_single(df, alignment_topic, should_flip=False):
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    if should_flip:
        ax[0].plot(df.index, df[alignment_topic], label="Bag")
        ax[0].plot(df.index, df[ulog_vo_y], label="Ulog")
        ax[1].plot(df.index, df[alignment_topic] - df[ulog_vo_y], label="Difference")
    else:
        ax[0].plot(df.index, df[alignment_topic], label="Bag")
        ax[0].plot(df.index, -df[ulog_vo_y], label="Ulog")
        ax[1].plot(df.index, df[alignment_topic] + df[ulog_vo_y], label="Difference")
    ax[0].legend()
    ax[1].legend()
    display(fig)
    plt.close()
    
def validate_alignment(dfs, alignment_topic, should_flip=False):
    for df in dfs:
        validate_alignment_single(df, alignment_topic, should_flip)
        
def create_pose_wrt_tar_single(df, static=True, vision=True, ulog=True, mavros=True):
    
    # if static, just use the target pose at plan time
    if static and vision:
        planned_row = df.loc[df["grasp_state_machine_node-grasp_started-data"].dropna().index[0]]
        
        est_quat = np.array([planned_row[mavros_gtsam_target_qx], planned_row[mavros_gtsam_target_qy],
                planned_row[mavros_gtsam_target_qz], planned_row[mavros_gtsam_target_qw]]).reshape(4,)
        tar_wrt_odom_R = R.from_quat(est_quat).as_matrix().T
        tar_wrt_odom_R[2,0] = 0
        tar_wrt_odom_R[2,1] = 0
        tar_wrt_odom_R[:,0] = tar_wrt_odom_R[:,0] / np.linalg.norm(tar_wrt_odom_R[:,0])
        tar_wrt_odom_R[:,1] = tar_wrt_odom_R[:,1] / np.linalg.norm(tar_wrt_odom_R[:,1])
        tar_wrt_odom_R[:,2] = [0,0,1]


        tar_wrt_odom_pos = np.array([planned_row[mavros_gtsam_target_x], planned_row[mavros_gtsam_target_y],
                                     planned_row[mavros_gtsam_target_z]])              
        tar_wrt_odom_vel = np.array([planned_row[mavros_gtsam_target_vx], planned_row[mavros_gtsam_target_vy],
                                     planned_row[mavros_gtsam_target_vz]])
        tar_wrt_odom_ang_vel = np.array([np.zeros_like(planned_row[mavros_gtsam_target_omega_z]), np.zeros_like(planned_row[mavros_gtsam_target_omega_z]),
                                     planned_row[mavros_gtsam_target_omega_z]])
    if not static:
        # Freeze target estimate after it picked up
        last_idx = len(df[:4])
        est_quat = np.array([df[mavros_gtsam_target_qx], df[mavros_gtsam_target_qy],
                df[mavros_gtsam_target_qz], df[mavros_gtsam_target_qw]]).T
        # Replace nans with identity quat to not break scipy
        nan_rows = np.argwhere(np.isnan(est_quat).any(axis=1))
        est_quat[nan_rows] = [0,0,0,1]
        tar_wrt_odom_R = R.from_quat(est_quat).as_matrix()
        tar_wrt_odom_R  = tar_wrt_odom_R.transpose(0,2,1)
        zero_col = tar_wrt_odom_R[:,:,0]
        zero_col[:,2] = 0
        zero_col = np.divide(zero_col.T, np.linalg.norm(zero_col, axis=1)).T
        first_col = tar_wrt_odom_R[:,:,1]
        first_col[:,2] = 0
        first_col = np.divide(first_col.T, np.linalg.norm(first_col, axis=1)).T
        tar_wrt_odom_R[:,:,0] = zero_col
        tar_wrt_odom_R[:,:,1] = first_col
        tar_wrt_odom_R[:,:,2] = [0,0,1]
        # Put the nans back in
        nan_arr = np.empty((3,3))
        nan_arr[:] = None
        tar_wrt_odom_R[nan_rows] = nan_arr
        tar_wrt_odom_R[last_idx:] = tar_wrt_odom_R[last_idx]
        
        tar_wrt_odom_pos = np.array([df[mavros_gtsam_target_x], df[mavros_gtsam_target_y],
                             df[mavros_gtsam_target_z]]).T
        tar_wrt_odom_pos[last_idx:] = tar_wrt_odom_pos[last_idx]
            
        tar_wrt_odom_vel = np.array([df[mavros_gtsam_target_vx], df[mavros_gtsam_target_vy],
                                     df[mavros_gtsam_target_vz]]).T
        tar_wrt_odom_vel[last_idx:] = tar_wrt_odom_vel[last_idx]
        
        tar_wrt_odom_ang_vel = np.array([np.zeros_like(df[mavros_gtsam_target_omega_z]), np.zeros_like(df[mavros_gtsam_target_omega_z]),
                                     df[mavros_gtsam_target_omega_z]]).T
        tar_wrt_odom_ang_vel[last_idx:] = tar_wrt_odom_ang_vel[last_idx]
        
    def transform_pos_vel(drone_wrt_odom_pos, drone_wrt_odom_vel):
        drone_wrt_tar_pos = np.subtract(drone_wrt_odom_pos, tar_wrt_odom_pos)
        drone_wrt_tar_vel = drone_wrt_odom_vel - tar_wrt_odom_vel - np.cross(tar_wrt_odom_ang_vel, drone_wrt_tar_pos)
#         print(tar_wrt_odom_R.shape)
#         print(drone_wrt_tar_vel.T.shape)
#         print(drone_wrt_tar_vel)

        drone_wrt_tar_pos = np.expand_dims(drone_wrt_tar_pos, axis=2)
        drone_wrt_tar_vel = np.expand_dims(drone_wrt_tar_vel, axis=2)
        print(tar_wrt_odom_R)
        return (tar_wrt_odom_R @ (drone_wrt_tar_pos)), (tar_wrt_odom_R @ (drone_wrt_tar_vel))
   
    if ulog:
        ulog_sp_wrt_odom_pos = np.array([df[ulog_sp_y], df[ulog_sp_x], -df[ulog_sp_z]]).T
        ulog_sp_wrt_odom_vel = np.array([df[ulog_sp_vy], df[ulog_sp_vx], -df[ulog_sp_vz]]).T
        ulog_sp_wrt_tar_pos, ulog_sp_wrt_tar_vel = transform_pos_vel(ulog_sp_wrt_odom_pos, ulog_sp_wrt_odom_vel)
        
        ulog_wrt_odom_pos = np.array([df[ulog_y], df[ulog_x], -df[ulog_z]]).T
        ulog_wrt_odom_vel = np.array([df[ulog_vy], df[ulog_vx], -df[ulog_vz]]).T
        ulog_wrt_tar_pos, ulog_wrt_tar_vel = transform_pos_vel(ulog_wrt_odom_pos, ulog_wrt_odom_vel)
        
#         print(ulog_sp_wrt_tar_pos.shape)
        df[ulog_sp_wrt_tar_x] = ulog_sp_wrt_tar_pos[:,0]
        df[ulog_sp_wrt_tar_y] = ulog_sp_wrt_tar_pos[:,1]
        df[ulog_sp_wrt_tar_z] = ulog_sp_wrt_tar_pos[:,2]

        df[ulog_sp_wrt_tar_vx] = ulog_sp_wrt_tar_vel[:,0]
        df[ulog_sp_wrt_tar_vy] = ulog_sp_wrt_tar_vel[:,1]
        df[ulog_sp_wrt_tar_vz] = ulog_sp_wrt_tar_vel[:,2]

        df[ulog_wrt_tar_x] = ulog_wrt_tar_pos[:,0]
        df[ulog_wrt_tar_y] = ulog_wrt_tar_pos[:,1]
        df[ulog_wrt_tar_z] = ulog_wrt_tar_pos[:,2]

        df[ulog_wrt_tar_vx] = ulog_wrt_tar_vel[:,0]
        df[ulog_wrt_tar_vy] = ulog_wrt_tar_vel[:,1]
        df[ulog_wrt_tar_vz] = ulog_wrt_tar_vel[:,2]
        
    if mavros:
        mavros_sp_wrt_odom_pos = np.array([df[mavros_drone_sp_x], df[mavros_drone_sp_y], df[mavros_drone_sp_z]]).T
        mavros_sp_wrt_odom_vel = np.array([df[mavros_drone_sp_vx], df[mavros_drone_sp_vy], df[mavros_drone_sp_vz]]).T
        mavros_sp_wrt_tar_pos, mavros_sp_wrt_tar_vel = transform_pos_vel(mavros_sp_wrt_odom_pos, mavros_sp_wrt_odom_vel)


        mavros_wrt_odom_pos = np.array([df[mavros_in_x], df[mavros_in_y], df[mavros_in_z]]).T
        mavros_wrt_odom_vel = np.array([df[mavros_in_vx], df[mavros_in_vy], df[mavros_in_vz]]).T
        mavros_wrt_tar_pos, mavros_wrt_tar_vel = transform_pos_vel(mavros_wrt_odom_pos, mavros_wrt_odom_vel)

        df[mavros_sp_wrt_tar_x] = mavros_sp_wrt_tar_pos[:,0]
        df[mavros_sp_wrt_tar_y] = mavros_sp_wrt_tar_pos[:,1]
        df[mavros_sp_wrt_tar_z] = mavros_sp_wrt_tar_pos[:,2]

        df[mavros_sp_wrt_tar_vx] = mavros_sp_wrt_tar_vel[:,0]
        df[mavros_sp_wrt_tar_vy] = mavros_sp_wrt_tar_vel[:,1]
        df[mavros_sp_wrt_tar_vz] = mavros_sp_wrt_tar_vel[:,2]

        df[mavros_wrt_tar_x] = mavros_wrt_tar_pos[:,0]
        df[mavros_wrt_tar_y] = mavros_wrt_tar_pos[:,1]
        df[mavros_wrt_tar_z] = mavros_wrt_tar_pos[:,2]

        df[mavros_wrt_tar_vx] = mavros_wrt_tar_vel[:,0]
        df[mavros_wrt_tar_vy] = mavros_wrt_tar_vel[:,1]
        df[mavros_wrt_tar_vz] = mavros_wrt_tar_vel[:,2]
    
def create_pose_wrt_tar(dfs, static=True, vision=True, ulog=True, mavros=True):
    for df in dfs:
        create_pose_wrt_tar_single(df, static=static, vision=vision, ulog=ulog, mavros=mavros)


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
        
def add_drone_velocity(df, dt=0.01):
    df["sparksdrone-world-pose.velocity.x"] = (df["sparksdrone-world-pose.position.x"].diff().rolling(20).mean()) / dt
    df["sparksdrone-world-pose.velocity.y"] = (df["sparksdrone-world-pose.position.y"].diff().rolling(20).mean()) / dt
    df["sparksdrone-world-pose.velocity.z"] = (df["sparksdrone-world-pose.position.z"].diff().rolling(20).mean()) / dt

def add_target_velocity(df, dt=0.01):
    df["sparkgrasptar-world-pose.velocity.x"] = (df["sparkgrasptar-world-pose.position.x"].diff().rolling(20).mean()) / dt
    df["sparkgrasptar-world-pose.velocity.y"] = (df["sparkgrasptar-world-pose.position.y"].diff().rolling(20).mean()) / dt
    df["sparkgrasptar-world-pose.velocity.z"] = (df["sparkgrasptar-world-pose.position.z"].diff().rolling(20).mean()) / dt

def add_drone_velocities(dfs):
    for df in dfs:
        add_drone_velocity(df)
        
def add_target_velocities(dfs):
    for df in dfs:
        add_target_velocity(df)
    


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
        display(fig)
        plt.close()
#         fig, ax = plt.subplots(1, 1, figsize=(15,10))
#         ax.plot(df.index, df[mavros_in_x], label="Marvros x")
#         ax.plot(df.index, df[mavros_in_y], label="Marvros y")
#         ax.plot(df.index, df[mavros_in_z], label="Marvros z")
        
#         ax.plot(df.index, df[mavros_drone_sp_x], label="Marvros sp x")
#         ax.plot(df.index, df[mavros_drone_sp_y], label="Marvros sp y")
#         ax.plot(df.index, df[mavros_drone_sp_z], label="Marvros sp z")
#         ax.legend()
#         ax.axvline(x=[df[df[gripper_state] == 0].index[0]], color='0.5', ls='--', lw=2)

def verify_alignment_mocap(dfs):
    for df in dfs:
        fig, ax = plt.subplots(1, 1, figsize=(15,10))
#         ax.plot(df.index, df[mocap_drone_x] - df[mavros_in_aligned_x], label="Mocap Drone x")
#         ax.plot(df.index, df[mocap_drone_y] - df[mavros_in_aligned_y], label="Mocap Drone y")
#         ax.plot(df.index, df[mocap_drone_z] - df[mavros_in_aligned_z], label="Mocap Drone z")

        ax.plot(df.index, df[mocap_aligned_x], label="Mocap aligned x")
        ax.plot(df.index, df[mocap_aligned_y], label="Mocap aligned y")
        ax.plot(df.index, df[mocap_aligned_z], label="Mocap aligned z")
        
        ax.plot(df.index, df[mavros_wrt_tar_x], label="Marvros x")
        ax.plot(df.index, df[mavros_wrt_tar_y], label="Marvros y")
        ax.plot(df.index, df[mavros_wrt_tar_z], label="Marvros z")
        
        ax.axvline(x=[df[df[gripper_state] == 0].index[0]], color='0.5', ls='--', lw=2)
        display(fig)
        plt.close()

def add_errors(dfs, start=-5, end=15, vision=True, ulog=True, mocap_target=True, mocap_drone=True, mavros=True):
    for df in dfs:
        if vision and mocap_drone:
            df = add_error_columns(df, [mavros_in_aligned_x, mavros_in_aligned_y, mavros_in_aligned_z],
                          [mocap_drone_x, mocap_drone_y, mocap_drone_z],
                          [alignment_error_x, alignment_error_y, alignment_error_z],
                           alignment_error_norm
                          )
            
            df = add_error_columns(df, [mocap_aligned_x, mocap_aligned_y, mocap_aligned_z],
              [mavros_wrt_tar_x, mavros_wrt_tar_y, mavros_wrt_tar_z],
              [mocap_alignment_error_x, mocap_alignment_error_y, mocap_alignment_error_z],
               mocap_alignment_error_norm
              )     

        if vision and mocap_target:
            df = add_error_columns(df, 
                          [mocap_gtsam_target_x, mocap_gtsam_target_y, mocap_gtsam_target_z],
                          [mocap_target_x, mocap_target_y, mocap_target_z],
                          [target_error_x, target_error_y, target_error_z],
                                   target_error_norm
                          )

        if ulog:    
            df = add_error_columns(df, [ulog_x, ulog_y, ulog_z],
                          [ulog_sp_x, ulog_sp_y, ulog_sp_z],
                          [ulog_error_x, ulog_error_y, ulog_error_z],
                           ulog_pos_error_norm
                          )
            
            df = add_error_columns(df, [ulog_wrt_tar_x, ulog_wrt_tar_y, ulog_wrt_tar_z],
                          [ulog_sp_wrt_tar_x, ulog_sp_wrt_tar_y, ulog_sp_wrt_tar_z],
                          [ulog_wrt_tar_error_x, ulog_wrt_tar_error_y, ulog_wrt_tar_error_z],
                           ulog_wrt_tar_pos_error_norm
                          )

            df = add_error_columns(df, [ulog_vx, ulog_vy, ulog_vz],
                          [ulog_sp_vx, ulog_sp_vy, ulog_sp_vz],
                          [ulog_error_vx, ulog_error_vy, ulog_error_vz],
                           ulog_vel_error_norm
                          )
            
            df = add_error_columns(df, [ulog_wrt_tar_vx, ulog_wrt_tar_vy, ulog_wrt_tar_vz],
                          [ulog_sp_wrt_tar_vx, ulog_sp_wrt_tar_vy, ulog_sp_wrt_tar_vz],
                          [ulog_wrt_tar_error_vx, ulog_wrt_tar_error_vy, ulog_wrt_tar_error_vz],
                           ulog_wrt_tar_vel_error_norm
                          )
                
        if mavros:    
            df = add_error_columns(df, [mavros_in_x, mavros_in_y, mavros_in_z],
                              [mavros_drone_sp_x, mavros_drone_sp_y, mavros_drone_sp_z],
                              [mavros_error_x, mavros_error_y, mavros_error_z],
                               mavros_pos_error_norm
                              )
            
            df = add_error_columns(df, [mavros_wrt_tar_x, mavros_wrt_tar_y, mavros_wrt_tar_z],
                              [mavros_sp_wrt_tar_x, mavros_sp_wrt_tar_y, mavros_sp_wrt_tar_z],
                              [mavros_wrt_tar_error_x, mavros_wrt_tar_error_y, mavros_wrt_tar_error_z],
                               mavros_wrt_tar_pos_error_norm
                              )

            df = add_error_columns(df, [mavros_in_vx, mavros_in_vy, mavros_in_vz],
                          [mavros_drone_sp_vx, mavros_drone_sp_vy, mavros_drone_sp_vz],
                          [mavros_error_vx, mavros_error_vy, mavros_error_vz],
                           mavros_vel_error_norm
                          )

            df = add_error_columns(df, [mavros_wrt_tar_vx, mavros_wrt_tar_vy, mavros_wrt_tar_vz],
                          [mavros_sp_wrt_tar_vx, mavros_sp_wrt_tar_vy, mavros_sp_wrt_tar_vz],
                          [mavros_wrt_tar_error_vx, mavros_wrt_tar_error_vy, mavros_wrt_tar_error_vz],
                           mavros_wrt_tar_vel_error_norm
                          )
            
    mean_df, std_df = get_aggregate_df(dfs) 
    return mean_df[start:end], std_df[start:end]


# +
def check_relative_pos(dfs, ulog=True):
    for df in dfs:
        sliced_df = df[-5:8]
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        if ulog:
            ax.plot(sliced_df[ulog_wrt_tar_x], label="Longitudinal")
            ax.plot(sliced_df[ulog_wrt_tar_y], label="Lateral")
            ax.plot(sliced_df[ulog_wrt_tar_z], label="Vertical")
        else:
            ax.plot(sliced_df[mavros_wrt_tar_x], label="Longitudinal")
            ax.plot(sliced_df[mavros_wrt_tar_y], label="Lateral")
            ax.plot(sliced_df[mavros_wrt_tar_z], label="Vertical")
            
#         plot_mean_std(ax, df, medkit_mean_df[ulog_wrt_tar_pos_error_norm], medkit_std_df[ulog_wrt_tar_pos_error_norm], label="Med-kit")
#         plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_pos_error_norm], pepsi_std_df[ulog_wrt_tar_pos_error_norm], label="Two-liter")
#         plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_pos_error_norm], cardboard_box_std_df[ulog_wrt_tar_pos_error_norm], label="Cardboard Box")
        ax.axvline(x=[0], color='0.5', ls='--', lw=2)
        ax.axvline(x=[4], color='0', ls='--', lw=2)
#         ax.set_xlim(0, 15)
#         ax.set_ylim(0, 0.25)
        ax.set_ylabel('Pos. [m]')
        ax.set_xlabel('Time [s]')
        # ax.set_xticklabels([])
        ax.legend()
    display(fig)
    plt.close()
    
def check_mean_and_std(dfs, mean_df, std_df, ulog=True):
    if ulog:
        topic = ulog_x
    else:
        topic = mavros_out_x
    sliced_df = dfs[0][-4:8]
    topic_xs = np.zeros((len(dfs), (len(sliced_df))))
    for i, df in enumerate(dfs):
        sliced_df = df[-4:8]
        topic_xs[i] = sliced_df[topic]
    mean = np.mean(topic_xs, axis=0)
    std = np.std(topic_xs, axis=0, ddof=1)
    mean_df_mean = mean_df[-4:8][topic]
    std_df_std = std_df[-4:8][topic]
    print(mean)
    print(mean_df_mean)
    print(np.allclose(mean, mean_df_mean))
    print(np.allclose(std, std_df_std))
    
def check_errors(dfs, ulog=True):
    for df in dfs:
        sliced_df = df[-4:8]
        if ulog:
            error = sliced_df[ulog_x] - sliced_df[ulog_sp_x]
            df_error = sliced_df[ulog_error_x]   
        else:
            error = sliced_df[mavros_in_x] - sliced_df[mavros_drone_sp_x]
            df_error = sliced_df[mavros_error_x]
        print(np.allclose(error, df_error))
        
def check_traj(dfs, ulog=True):
    for df in dfs:
        sliced_df = df[-5:8]
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        if ulog:
            ax.plot(sliced_df[ulog_x], label="Ulog x")
            ax.plot(sliced_df[ulog_y], label="Ulog y")
            ax.plot(sliced_df[ulog_z], label="Ulog z")
        else:
            ax.plot(sliced_df[mavros_in_x], label="mavros x")
            ax.plot(sliced_df[mavros_in_y], label="mavros y")
            ax.plot(sliced_df[mavros_in_z], label="mavros z")
        ax.axvline(x=[0], color='0.5', ls='--', lw=2)
        ax.axvline(x=[4], color='0', ls='--', lw=2)
#         ax.set_xlim(0, 15)
#         ax.set_ylim(0, 0.25)
        ax.set_ylabel('Pos. [m]')
        ax.set_xlabel('Time [s]')
        # ax.set_xticklabels([])
        ax.legend()
        display(fig)
        plt.close()
    
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        if ulog:
            ax.plot(sliced_df[ulog_vx], label="Ulog vx")
            ax.plot(sliced_df[ulog_vy], label="Ulog vy")
            ax.plot(sliced_df[ulog_vz], label="Ulog vz")
        else:
            ax.plot(sliced_df[mavros_in_vx], label="mavros vx")
            ax.plot(sliced_df[mavros_in_vy], label="mavros vy")
            ax.plot(sliced_df[mavros_in_vz], label="mavros vz")
        ax.axvline(x=[0], color='0.5', ls='--', lw=2)
        ax.axvline(x=[4], color='0', ls='--', lw=2)
#         ax.set_xlim(0, 15)
#         ax.set_ylim(0, 0.25)
        ax.set_ylabel('Vel. [m]')
        ax.set_xlabel('Time [s]')
        # ax.set_xticklabels([])
        ax.legend()
        display(fig)
        plt.close()

def check_velocity(dfs, ulog=True):
    for df in dfs:
        sliced_df = df[-5:8]
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        if ulog:
            ax.plot(sliced_df[ulog_vx], label="Ulog vx")
            ax.plot(sliced_df[ulog_vy], label="Ulog vy")
            ax.plot(sliced_df[ulog_vz], label="Ulog vz")
        else:
            ax.plot(sliced_df[mavros_in_vx], label="Ulog vx")
            ax.plot(sliced_df[mavros_in_vy], label="Ulog vy")
            ax.plot(sliced_df[mavros_in_vz], label="Ulog vz")
            
        
        ax.plot(sliced_df[mocap_drone_vy], label="Mocap vx")
        ax.plot(sliced_df[mocap_drone_vx], label="Mocap vy")
        ax.plot(-sliced_df[mocap_drone_vz], label="Mocap vz")
        ax.axvline(x=[0], color='0.5', ls='--', lw=2)
        ax.axvline(x=[4], color='0', ls='--', lw=2)
        ax.set_ylabel('Vel. [m]')
        ax.set_xlabel('Time [s]')
        # ax.set_xticklabels([])
        ax.legend()
        display(fig)
        plt.close()
        
def check_grasp_velocity(dfs, ulog=True, topic=mocap_drone_x):
    for df in dfs:
        start = df.loc[[3.5]][topic].values
        end = df.loc[[4.5]][topic].values
#         print(start.values)
        print( (end- start) / 1)
    
def check_relative_error(mean_df, ulog=True):
    sliced_df = mean_df[-5:8]
    if ulog:
        print(np.allclose(sliced_df[ulog_wrt_tar_pos_error_norm], sliced_df[ulog_pos_error_norm], atol=1e-2))
        print(np.allclose(sliced_df[ulog_wrt_tar_vel_error_norm], sliced_df[ulog_vel_error_norm], atol=1e-2))
    else:
        print(np.allclose(sliced_df[mavros_wrt_tar_pos_error_norm], sliced_df[mavros_pos_error_norm], atol=1e-2))
        print(np.allclose(sliced_df[mavros_wrt_tar_vel_error_norm], sliced_df[mavros_vel_error_norm], atol=1e-2))

    
    
def confirm_accuracy(dfs, mean_df, std_df, alignment_topic=None, should_flip=False, mavros=True,
                     name=None, ulog=True, mocap=True, mocap_grasp_axis=mocap_drone_x):
    print("Confirming accuracy for {}...".format(name))
    if ulog:
        print("Checking that the bags are aligned to ulogs...")
        validate_alignment(dfs, alignment_topic, should_flip=should_flip)
    if mocap and mavros:
        print("Check that mavros aligned to mocap looks reasonable...")
        verify_alignment(dfs)
        print("Check that mocap aligned to mavros looks reasonable...")
        verify_alignment_mocap(dfs)
    print("Check that mean and std looks right...")
    check_mean_and_std(dfs, mean_df, std_df, ulog=ulog)
    print("Check errors...")
    check_errors(dfs, ulog=ulog)
    print("Check traj shape...")
    check_traj(dfs, ulog=ulog)
    if mocap:
        print("Compare estimated velocity with mocap finite difference...")
        check_velocity(dfs, ulog=ulog)
        print("Check grasp velocity with straight line derivative...")
        check_grasp_velocity(dfs, topic=mocap_grasp_axis)
    print("Check that normal and relative target errors are identical...")
    check_relative_error(mean_df, ulog=ulog)
    
    

#     print("Check that pos wrt to tar is basically same as global pos...")
#     check_relative_pos(dfs)
    
    
# -

# # Static Plots

# Load all the data

# +
ulog_location = "../FinalPass/vision_medkit_05ms"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/vision_medkit_05ms"
bag_result_location = "../FinalPass/vision_medkit_05ms"

medkit_dfs = create_dfs(ulog_location, ulog_result_location, 
                        bag_location, bag_result_location, 
                        alignment_topic=mavros_out_x)
validate_alignment(medkit_dfs, mavros_out_x)
create_aligned_mavros(medkit_dfs)
create_pose_wrt_tar(medkit_dfs)
create_aligned_mocap(medkit_dfs)

ulog_location = "../FinalPass/vision_cardboard_box_05ms"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/vision_cardboard_box_05ms"
bag_result_location = "../FinalPass/vision_cardboard_box_05ms"

cardboard_box_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_out_x)
del cardboard_box_dfs[9]
validate_alignment(cardboard_box_dfs, mavros_out_x)
create_aligned_mavros(cardboard_box_dfs)
create_pose_wrt_tar(cardboard_box_dfs)
create_aligned_mocap(cardboard_box_dfs)

ulog_location = "../FinalPass/vision_pepsi_05ms"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/vision_pepsi_05ms"
bag_result_location = "../FinalPass/vision_pepsi_05ms"

pepsi_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_out_x)
validate_alignment(pepsi_dfs, mavros_out_x)
create_aligned_mavros(pepsi_dfs)
create_pose_wrt_tar(pepsi_dfs)
create_aligned_mocap(pepsi_dfs)

ulog_location = "../FinalPass/vision_pepsi_125ms"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/vision_pepsi_125ms"
bag_result_location = "../FinalPass/vision_pepsi_125ms"

pepsi_mid_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_out_x)
del pepsi_mid_dfs[10]
validate_alignment(pepsi_mid_dfs, mavros_out_x)
create_aligned_mavros(pepsi_mid_dfs)
create_pose_wrt_tar(pepsi_mid_dfs)
create_aligned_mocap(pepsi_mid_dfs)

ulog_location = "../FinalPass/vision_pepsi_2ms"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/vision_pepsi_2ms"
bag_result_location = "../FinalPass/vision_pepsi_2ms"

pepsi_fast_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mavros_out_x)
del pepsi_fast_dfs[10]
validate_alignment(pepsi_fast_dfs, mavros_out_x)
create_aligned_mavros(pepsi_fast_dfs)
create_pose_wrt_tar(pepsi_fast_dfs)
create_aligned_mocap(pepsi_fast_dfs)

ulog_location = None
ulog_result_location = None
bag_location = "../FinalPass/vision_pepsi_3ms"
bag_result_location = "../FinalPass/vision_pepsi_3ms"

pepsi_fastest_dfs = create_dfs(ulog_location, ulog_result_location, 
                               bag_location, bag_result_location)
# Omit catastrophic runs
del pepsi_fastest_dfs[7]

create_aligned_mavros(pepsi_fastest_dfs)
create_pose_wrt_tar(pepsi_fastest_dfs, ulog=False)
create_aligned_mocap(pepsi_fastest_dfs)

ulog_location = None
ulog_result_location = None
bag_location = "../FinalPass/vision_medkit_outdoors"
bag_result_location = "../FinalPass/vision_medkit_outdoors"

outdoors_dfs = create_dfs(ulog_location, ulog_result_location, 
                               bag_location, bag_result_location)
# Omit catastrophic runs
del outdoors_dfs[5]
del outdoors_dfs[7]
del outdoors_dfs[8]

create_pose_wrt_tar(outdoors_dfs, ulog=False)

# -

add_drone_velocities(medkit_dfs)
add_drone_velocities(cardboard_box_dfs)
add_drone_velocities(pepsi_dfs)
add_drone_velocities(pepsi_mid_dfs)
add_drone_velocities(pepsi_fast_dfs)
add_drone_velocities(pepsi_fastest_dfs)

medkit_mean_df, medkit_std_df = add_errors(medkit_dfs)
cardboard_box_mean_df, cardboard_box_std_df = add_errors(cardboard_box_dfs)
pepsi_mean_df, pepsi_std_df = add_errors(pepsi_dfs)
pepsi_mid_mean_df, pepsi_mid_std_df = add_errors(pepsi_mid_dfs)
pepsi_fast_mean_df, pepsi_fast_std_df = add_errors(pepsi_fast_dfs)
pepsi_fastest_mean_df, pepsi_fastest_std_df = add_errors(pepsi_fastest_dfs, ulog=False, mocap_target=False)
outdoors_mean_df, outdoors_std_df = add_errors(outdoors_dfs, ulog=False, mocap_target=False, mocap_drone=False)

confirm_accuracy(medkit_dfs, medkit_mean_df, medkit_std_df, alignment_topic=mavros_out_x, name="Vision Medkit 0.5 m/s")

confirm_accuracy(cardboard_box_dfs, cardboard_box_mean_df, cardboard_box_std_df, alignment_topic=mavros_out_x, name="Vision Cardboard Box 0.5 m/s")

confirm_accuracy(pepsi_dfs, pepsi_mean_df, pepsi_std_df, alignment_topic=mavros_out_x, name="Vision Pepsi 0.5 m/s")

confirm_accuracy(pepsi_mid_dfs, pepsi_mid_mean_df, pepsi_mid_std_df, alignment_topic=mavros_out_x, name="Vision Pepsi 1.25 m/s")

confirm_accuracy(pepsi_fast_dfs, pepsi_fast_mean_df, pepsi_fast_std_df, alignment_topic=mavros_out_x, name="Vision Pepsi 2 m/s")

confirm_accuracy(pepsi_fastest_dfs, pepsi_fastest_mean_df, pepsi_fastest_std_df, ulog=False, alignment_topic=mavros_out_x, name="Vision Pepsi 3 m/s")

confirm_accuracy(outdoors_dfs, outdoors_mean_df, outdoors_std_df, ulog=False, mocap=False, name="Vision Outdoors 0.5 m/s")

# +
# Objects position and velocity tracking errors:
fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_pos_error_norm], medkit_std_df[ulog_wrt_tar_pos_error_norm], label="Med-kit")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_pos_error_norm], pepsi_std_df[ulog_wrt_tar_pos_error_norm], label="Two-liter")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_pos_error_norm], cardboard_box_std_df[ulog_wrt_tar_pos_error_norm], label="Cardboard Box")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylim(0, 0.25)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('object_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_vel_error_norm], medkit_std_df[ulog_wrt_tar_vel_error_norm], label="Med-kit")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_vel_error_norm], pepsi_std_df[ulog_wrt_tar_vel_error_norm], label="Two-liter")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_vel_error_norm], cardboard_box_std_df[ulog_wrt_tar_vel_error_norm], label="Cardboard Box")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylim(0, 0.5)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('object_vel_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_pos_error_norm], pepsi_std_df[ulog_wrt_tar_pos_error_norm], label="0.5 m/s")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_pos_error_norm], pepsi_mid_std_df[ulog_wrt_tar_pos_error_norm], label="1.25 m/s")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_pos_error_norm], pepsi_fast_std_df[ulog_wrt_tar_pos_error_norm], label="2 m/s")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_wrt_tar_pos_error_norm], pepsi_fastest_std_df[mavros_wrt_tar_pos_error_norm], label="3 m/s")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_xlim(0, 15)
ax.legend()
fig.savefig('speed_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_vel_error_norm], pepsi_std_df[ulog_wrt_tar_vel_error_norm], label="0.5 m/s")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_vel_error_norm], pepsi_mid_std_df[ulog_wrt_tar_vel_error_norm], label="1.25 m/s")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_vel_error_norm], pepsi_fast_std_df[ulog_wrt_tar_vel_error_norm], label="2 m/s")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_wrt_tar_vel_error_norm], pepsi_fastest_std_df[mavros_wrt_tar_vel_error_norm], label="3 m/s")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_xlim(0, 15)
ax.legend()
fig.savefig('speed_vel_err.svg')


# +
# Success Bar Plot
rates = {"vision_medkit_05" : 9,
         "mocap_medkit_05" : 7,
         "vision_medkit_outdoors" : 8,
         "vision_cardboard_05" : 6,
         "mocap_cardboard_05" : 8,
         "vision_pepsi_05" : 10,
         "mocap_pepsi_05" : 10,
         "vision_pepsi_125" : 10,
         "mocap_pepsi_125" : 9,
         "vision_pepsi_2" : 7,
         "mocap_pepsi_2" : 6,
         "vision_pepsi_3" : 3,
         "mocap_pepsi_3" : 4}


vision = (rates["vision_medkit_05"], 
          rates["vision_cardboard_05"], 
          rates["vision_pepsi_05"],
          rates["vision_pepsi_125"],
          rates["vision_pepsi_2"],
          rates["vision_pepsi_3"])
                
mocap =  (rates["mocap_medkit_05"], 
          rates["mocap_cardboard_05"], 
          rates["mocap_pepsi_05"],
          rates["mocap_pepsi_125"],
          rates["mocap_pepsi_2"],
          rates["mocap_pepsi_3"])

fig, ax = plt.subplots(1,1, figsize=(12,6))

ind = np.arange(6) + 1
width = 0.3

ax.bar(ind, vision, width, label='Vision')
ax.bar(ind + width, mocap, width, label='Mocap')
ax.bar(.15, rates["vision_medkit_outdoors"], width, color="b")
ax.set_ylabel('Success')

all_ind = np.arange(7)
ax.set_xticks(all_ind + width / 2, ('Outdoors', 'Med-kit', 'Cardboard\n Box', 'Two-liter', '1.25 m/s', '2 m/s', '3 m/s'))
ax.legend(loc='best')
ax.axhline(y=10, color='0', ls='--', lw=2)
fig.savefig('static_success.svg')


# +
# Pose Estimate Table
def get_pose_error(planned_row):
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
    # We want to express error in estimated target frame, a little confusing, its probably right,
    # but it also doesn't really make a difference because the frames are basically aligned.
    error_t = est_R.T.dot(est_t - gt_t)
    error_t_norm = np.linalg.norm(error_t)
    
    proj_est_x = [est_R[0,0], est_R[0,1]] / np.linalg.norm([est_R[0,0], est_R[0,1]])
    proj_gt_x = [gt_R[0,0], gt_R[0,1]] / np.linalg.norm([gt_R[0,0], gt_R[0,1]])
    error_yaw = np.rad2deg(np.arccos((proj_est_x.dot(proj_gt_x))))
    
    return error_R, error_t, error_t_norm, error_yaw

# VIO Drift Table
def get_vio_error(df):
    df_sliced = df[-5:4]
    vio_error_x = np.mean(np.abs(df_sliced[alignment_error_x]))
    vio_error_y = np.mean(np.abs(df_sliced[alignment_error_y]))
    vio_error_z = np.mean(np.abs(df_sliced[alignment_error_z]))
    vio_error = np.array([vio_error_x, vio_error_y, vio_error_z])
    vio_error_norm = np.mean(df_sliced[alignment_error_norm])
    
    return vio_error, vio_error_norm

# This is like this so that the vio error is expressed wrt target. It doesnt really matter though, especially
# bc im just taking abs
def get_mocap_vio_error(df):
    df_sliced = df[-5:4]
    vio_error_x = np.mean(np.abs(df_sliced[mocap_alignment_error_x]))
    vio_error_y = np.mean(np.abs(df_sliced[mocap_alignment_error_y]))
    vio_error_z = np.mean(np.abs(df_sliced[mocap_alignment_error_z]))
    vio_error = np.array([vio_error_x, vio_error_y, vio_error_z])
    vio_error_norm = np.mean(df_sliced[mocap_alignment_error_norm])
    
    return vio_error, vio_error_norm

def get_speed_at_grasp(mean_df):
    return mean_df[mocap_drone_vx][4]

def get_tracking_error(df, ulog=True):
    if ulog:
        tracking_error = np.array([df[ulog_wrt_tar_error_x][4], 
                                          df[ulog_wrt_tar_error_y][4],
                                          df[ulog_wrt_tar_error_z][4]])
    else:
        tracking_error = np.array([df[mavros_wrt_tar_error_x][4], 
                                          df[mavros_wrt_tar_error_y][4],
                                          df[mavros_wrt_tar_error_z][4]])
    return tracking_error, np.abs(tracking_error)


def get_all_errors(dfs, ulog=True, target=True):
    avg_error_R = np.zeros((len(dfs), 1))
    avg_error_t = np.zeros((len(dfs), 3))
    abs_avg_error_t = np.zeros((len(dfs), 3))
    avg_error_t_norm = np.zeros((len(dfs), 1))
    avg_error_yaw = np.zeros((len(dfs), 1))
    avg_vio_error = np.zeros((len(dfs), 3))
    avg_vio_error_norm = np.zeros((len(dfs), 1))
    mocap_avg_vio_error = np.zeros((len(dfs), 3))
    mocap_avg_vio_error_norm = np.zeros((len(dfs), 1))
    avg_tracking_error = np.zeros((len(dfs), 3))
    abs_avg_tracking_error = np.zeros((len(dfs), 3))

    for i, df in enumerate(dfs):
        planned_row = df.loc[df["grasp_state_machine_node-grasp_started-data"].dropna().index[0]]
        if target:
            error_R, error_t, error_t_norm, error_yaw = get_pose_error(planned_row)
            avg_error_R[i] = error_R
            avg_error_t[i] = error_t
            abs_avg_error_t[i] = abs(error_t)
            avg_error_t_norm[i] = error_t_norm
            avg_error_yaw[i] = error_yaw
            
            
        vio_error, vio_error_norm = get_vio_error(df)
        mocap_vio_error, mocap_vio_error_norm = get_mocap_vio_error(df)
        tracking_error, abs_tracking_error = get_tracking_error(df, ulog)

        avg_vio_error[i] = vio_error
        avg_vio_error_norm[i] = vio_error_norm
        mocap_avg_vio_error[i] = mocap_vio_error
        mocap_avg_vio_error_norm[i] = mocap_vio_error_norm
        avg_tracking_error[i] = tracking_error
        abs_avg_tracking_error[i] = abs_tracking_error
    
#     avg_error_R /= len(dfs)
#     avg_error_t /= len(dfs)
# #     abs_avg_error_t /= len(dfs)
#     avg_error_t_norm /= len(dfs)
#     avg_error_yaw /= len(dfs)
#     avg_vio_error /= len(dfs)
#     avg_vio_error_norm /= len(dfs)
#     mocap_avg_vio_error /= len(dfs)
#     mocap_avg_vio_error_norm /= len(dfs)
#     avg_tracking_error /= len(dfs)
#     abs_avg_tracking_error /= len(dfs)
    actual_speed = get_speed_at_grasp(pepsi_mean_df)
    
    

    return avg_error_R, avg_error_t, abs_avg_error_t, \
           avg_error_t_norm, avg_error_yaw, avg_vio_error,\
           avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
           actual_speed, avg_tracking_error, abs_avg_tracking_error

def print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error):
#     print("Average Rotation Error: {}".format(np.mean(avg_error_R)))
#     print("STD Rotation Error: {}".format(np.std(avg_error_R)))
#     print("Average Translation Error: {}".format(np.mean(avg_error_t, axis=0)))
#     print("STD Translation Error: {}".format(np.std(avg_error_t, axis=0)))
    print("{} - Average Abs Translation Error".format(100 * np.mean(abs_avg_error_t, axis=0)))
    print("{} - STD Abs Translation Error".format(100 * np.std(abs_avg_error_t, axis=0)))
#     print("Average Translation Error Norm: {}".format(np.mean(avg_error_t_norm)))
#     print("STD Translation Error Norm: {}".format(np.std(avg_error_t_norm)))
#     print("Average Yaw Error: {}".format(np.mean(avg_error_yaw)))
#     print("STD Yaw Error: {}".format(np.std(avg_error_yaw)))
#     print("Average VIO Error: {}".format(np.mean(avg_vio_error, axis=0)))
#     print("STD VIO Error: {}".format(np.mean(avg_vio_error, axis=0)))
#     print("Average VIO Error Norm: {}".format(np.mean(avg_vio_error_norm)))
#     print("STD VIO Error Norm: {}".format(np.mean(avg_vio_error_norm)))
    print("{} - Average Mocap to Mavros VIO Error".format(100 * np.mean(mocap_avg_vio_error,axis=0)))
    print("{} - STD Mocap to Mavros VIO Error".format(100 * np.std(mocap_avg_vio_error,axis=0)))
#     print("Average Mocap to Mavros VIO Error Norm: {}".format(np.mean(mocap_avg_vio_error_norm)))
#     print("STD Mocap to Mavros VIO Error Norm: {}".format(np.std(mocap_avg_vio_error_norm)))
#     print("Avergae tracking error: {}".format(np.mean(avg_tracking_error,axis=0)))
#     print("STD tracking error: {}".format(np.std(avg_tracking_error,axis=0)))
    print("{} - Average Abs Tracking Error". format(100 * np.mean(abs_avg_tracking_error,axis=0)))
    print("{} - STD Abs Tracking Error". format(100 *np.std(abs_avg_tracking_error,axis=0)))  
#     print("Actual Speed: {}".format(actual_speed))
#     Average case, assume vio just points in the worst direction because the metric is already less clear
    average_case = np.mean(avg_tracking_error, axis=0) + np.mean(avg_error_t, axis=0)
    worst_vio = np.sign(average_case)*np.abs(np.mean(mocap_avg_vio_error,axis=0))
    average_case += worst_vio
#     print("Average case: {}".format(average_case))
#     print(np.std(avg_tracking_error, axis=0))
#     std = np.sqrt(np.std(avg_tracking_error, axis=0)**2 + np.std(avg_error_t, axis=0)**2 + np.std(worst_vio)**2)
#     print("Average case std: {}".format(std))
    print("{} - Worst case".format(100 * (np.mean(abs_avg_error_t, axis=0) + np.mean(mocap_avg_vio_error, axis=0) + np.mean(abs_avg_tracking_error,axis=0))))
    std = 100 * np.sqrt(np.std(abs_avg_tracking_error, axis=0)**2 + np.std(abs_avg_error_t, axis=0)**2 + np.std(mocap_avg_vio_error)**2)
    print("{} - Worst case std".format(std))
    print("{} - Effective Worst case".format(100 * (np.mean(abs_avg_error_t, axis=0) + np.mean(mocap_avg_vio_error, axis=0))))
    std = 100 * np.sqrt(np.std(abs_avg_error_t, axis=0)**2 + np.std(mocap_avg_vio_error, axis=0)**2)
    print("{} - Worst case std".format(std))
#     print("worst case 2: {}".format(np.abs(avg_error_t) +np.abs(mocap_avg_vio_error) + np.abs(avg_tracking_error)))
    print("-------------------------------------------")
    
    


print("Medkit Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(medkit_dfs, ulog=True)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)



print("Pepsi Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(pepsi_dfs, ulog=True)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)

print("Cardboard Box Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(cardboard_box_dfs, ulog=True)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)

print("Pepsi 1.25 Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(pepsi_mid_dfs, ulog=True)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)

print("Pepsi 2 Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(pepsi_fast_dfs, ulog=True)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)

print("Pepsi 3 Metrics:")
avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
actual_speed, avg_tracking_error, abs_avg_tracking_error = get_all_errors(pepsi_fastest_dfs, ulog=False, target=False)

print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
                  avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
                  actual_speed, avg_tracking_error, abs_avg_tracking_error)

# print("Outdoors Metrics:")
# avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
# avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
# actual_speed, abs_avg_tracking_error = get_all_errors(outdoors_dfs, ulog=False)

# print_metrics(avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw,\
#                   avg_vio_error, avg_vio_error_norm, mocap_avg_vio_error, mocap_avg_vio_error_norm,\
#                   actual_speed, abs_avg_tracking_error)



# avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw = get_all_errors(pepsi_dfs)

# print("Pepsi Metrics:")
# print("Average Rotation Error: {}".format(avg_error_R))
# print("Average Translation Error: {}".format(avg_error_t))
# print("Average Translation Error Norm: {}".format(avg_error_t_norm))
# print("Average Abs Translation Error: {}".format(abs_avg_error_t))
# print("Average Yaw Error: {}".format(avg_error_yaw))
# print("-------------------------------------------")

# avg_error_R, avg_error_t, abs_avg_error_t, avg_error_t_norm, avg_error_yaw = get_all_errors(cardboard_box_dfs)
# print("Cardboard Box Metrics:")
# print("Average Rotation Error: {}".format(avg_error_R))
# print("Average Translation Error: {}".format(avg_error_t))
# print("Average Translation Error Norm: {}".format(avg_error_t_norm))
# print("Average Abs Translation Error: {}".format(abs_avg_error_t))
# print("Average Yaw Error: {}".format(avg_error_yaw))
# print("-------------------------------------------")
    
    

# +
# VIO Drift Table
def get_vio_error(df):
    df_sliced = df[-5:4]
    vio_error_x = np.mean(np.abs(df_sliced[alignment_error_x]))
    vio_error_y = np.mean(np.abs(df_sliced[alignment_error_y]))
    vio_error_z = np.mean(np.abs(df_sliced[alignment_error_z]))
    vio_error = np.array([vio_error_x, vio_error_y, vio_error_z])
    vio_error_norm = np.mean(df_sliced[alignment_error_norm])
    
    return vio_error, vio_error_norm

def get_mocap_vio_error(df):
    df_sliced = df[-5:4]
    vio_error_x = np.mean(np.abs(df_sliced[mocap_alignment_error_x]))
    vio_error_y = np.mean(np.abs(df_sliced[mocap_alignment_error_y]))
    vio_error_z = np.mean(np.abs(df_sliced[mocap_alignment_error_z]))
    vio_error = np.array([vio_error_x, vio_error_y, vio_error_z])
    vio_error_norm = np.mean(df_sliced[mocap_alignment_error_norm])
    
    return vio_error, vio_error_norm

def get_speed_at_grasp(mean_df):
    return mean_df[mocap_drone_vx][4]
    
def get_table_entry(dfs, mean_df):
    avg_vio_error = np.array([0.,0.,0.])
    avg_vio_error_norm = 0
    mocap_avg_vio_error = np.array([0.,0.,0.])
    mocap_avg_vio_error_norm = 0
    for df in dfs:
        vio_error, vio_error_norm = get_vio_error(df)
        mocap_vio_error, mocap_vio_error_norm = get_mocap_vio_error(df)
        avg_vio_error += vio_error
        avg_vio_error_norm += vio_error_norm
        mocap_avg_vio_error += mocap_vio_error
        mocap_avg_vio_error_norm += mocap_vio_error_norm

    avg_vio_error /= len(dfs)
    avg_vio_error_norm /= len(dfs)
    mocap_avg_vio_error /= len(dfs)
    mocap_avg_vio_error_norm /= len(dfs)
    actual_speed = get_speed_at_grasp(mean_df)
    print("Average VIO Error: {}".format(avg_vio_error))
    print("Average VIO Error Norm: {}".format(avg_vio_error_norm))
    print("Average Mocap to Mavros VIO Error: {}".format(mocap_avg_vio_error))
    print("Average Mocap to Mavros VIO Error Norm: {}".format(mocap_avg_vio_error_norm))
    print("Actual Speed: {}".format(actual_speed))
    print("-------------------------------------------")


print("Pepsi 0.5 m/s Metrics:")
get_table_entry(pepsi_dfs, pepsi_mean_df)

print("Pepsi 1.25 m/s Metrics:")
get_table_entry(pepsi_mid_dfs, pepsi_mid_mean_df)

print("Pepsi 2 m/s Metrics:")
get_table_entry(pepsi_fast_dfs, pepsi_fast_mean_df)

print("Pepsi 3 m/s Metrics:")
get_table_entry(pepsi_fastest_dfs, pepsi_fastest_mean_df)

# -

# # Moving Plots

# +
# ulog_location = "../FinalPass/mocap_a1_slow"
# ulog_result_location = "../FinalPass/log_output"
# bag_location = "../FinalPass/mocap_a1_slow"
# bag_result_location = "../FinalPass/mocap_a1_slow"

# a1_slow_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mocap_drone_x,
#                  should_flip=True)
# validate_alignment(a1_slow_dfs, mocap_drone_x, should_flip=True)
# del a1_slow_dfs[4]
# create_pose_wrt_tar(a1_slow_dfs, static=False, mavros=False)


ulog_location = "../FinalPass/mocap_a1_fast"
ulog_result_location = "../FinalPass/log_output"
bag_location = "../FinalPass/mocap_a1_fast"
bag_result_location = "../FinalPass/mocap_a1_fast"

a1_fast_dfs = create_dfs(ulog_location, ulog_result_location, 
                 bag_location, bag_result_location, 
                 alignment_topic=mocap_drone_x,
                 should_flip=True)
validate_alignment(a1_fast_dfs, mocap_drone_x, should_flip=True)
create_pose_wrt_tar(a1_fast_dfs, static=False, mavros=False)
del a1_fast_dfs[10]
del a1_fast_dfs[11]


# ulog_location = "../FinalPass/mocap_turntable_05ms"
# ulog_result_location = "../FinalPass/log_output"
# bag_location = "../FinalPass/mocap_turntable_05ms"
# bag_result_location = "../FinalPass/mocap_turntable_05ms"

# turn_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mocap_drone_x,
#                  should_flip=True)
# validate_alignment(turn_dfs, mocap_drone_x, should_flip=True)
# create_pose_wrt_tar(turn_dfs, static=False, mavros=False)


# ulog_location = "../FinalPass/mocap_turntable_1ms"
# ulog_result_location = "../FinalPass/log_output"
# bag_location = "../FinalPass/mocap_turntable_1ms"
# bag_result_location = "../FinalPass/mocap_turntable_1ms"

# turn_mid_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mocap_drone_x,
#                  should_flip=True)
# validate_alignment(turn_mid_dfs, mocap_drone_x, should_flip=True)
# create_pose_wrt_tar(turn_mid_dfs, static=False, mavros=False)


# ulog_location = "../FinalPass/mocap_turntable_15ms"
# ulog_result_location = "../FinalPass/log_output"
# bag_location = "../FinalPass/mocap_turntable_15ms"
# bag_result_location = "../FinalPasss/mocap_turntable_15ms"

# turn_fast_dfs = create_dfs(ulog_location, ulog_result_location, 
#                  bag_location, bag_result_location, 
#                  alignment_topic=mocap_drone_x,
#                  should_flip=True)
# validate_alignment(turn_fast_dfs, mocap_drone_x, should_flip=True)
# create_pose_wrt_tar(turn_fast_dfs, static=False, mavros=False)



# +
add_drone_velocities(a1_slow_dfs)
add_drone_velocities(a1_fast_dfs)
add_drone_velocities(turn_dfs)
add_drone_velocities(turn_mid_dfs)
add_drone_velocities(turn_fast_dfs)

add_target_velocities(a1_slow_dfs)
add_target_velocities(a1_fast_dfs)
add_target_velocities(turn_dfs)
add_target_velocities(turn_mid_dfs)
add_target_velocities(turn_fast_dfs)
# -

odds_turn_dfs = [turn_dfs[i] for i in range(1, len(turn_dfs), 2)]
odds_turn_mid_dfs = [turn_mid_dfs[i] for i in range(1, len(turn_mid_dfs), 2)]
odds_turn_fast_dfs = [turn_fast_dfs[i] for i in range(1, len(turn_fast_dfs), 2)]

a1_slow_mean_df, a1_slow_std_df = add_errors(a1_slow_dfs, ulog=True, vision=False, mavros=False)
a1_fast_mean_df, a1_fast_std_df = add_errors(a1_fast_dfs, ulog=True, vision=False, mavros=False)
odds_turn_mean_df, odds_turn_std_df = add_errors(odds_turn_dfs, ulog=True, vision=False, mavros=False)
odds_turn_mid_mean_df, odds_turn_mid_std_df = add_errors(odds_turn_mid_dfs, ulog=True, vision=False, mavros=False)
odds_turn_fast_mean_df, odds_turn_fast_std_df = add_errors(odds_turn_fast_dfs, ulog=True, vision=False, mavros=False)

confirm_accuracy(a1_slow_dfs, a1_slow_mean_df, a1_slow_std_df, alignment_topic=mocap_drone_x, should_flip=True, mavros=False, mocap_grasp_axis=mocap_drone_y, name="Mocap A1 Slow")

confirm_accuracy(a1_fast_dfs, a1_fast_mean_df, a1_fast_std_df, alignment_topic=mocap_drone_x, should_flip=True, mavros=False, mocap_grasp_axis=mocap_drone_y, name="Mocap A1 Fast")

confirm_accuracy(odds_turn_dfs, odds_turn_mean_df, odds_turn_std_df, alignment_topic=mocap_drone_x, should_flip=True, mavros=False, mocap_grasp_axis=mocap_drone_x, name="Turntable Slow")

confirm_accuracy(odds_turn_mid_dfs, odds_turn_mid_mean_df, odds_turn_mid_std_df, alignment_topic=mocap_drone_x, should_flip=True, mavros=False, mocap_grasp_axis=mocap_drone_x, name="Turntable Mid")

confirm_accuracy(odds_turn_fast_dfs, odds_turn_fast_mean_df, odds_turn_fast_std_df, alignment_topic=mocap_drone_x, should_flip=True, mavros=False, mocap_grasp_axis=mocap_drone_x, name="Turntable Fast")

print("A1 slow mean", 100 * get_tracking_error(a1_slow_mean_df)[1])
print("A1 slow std", 100 * get_tracking_error(a1_slow_std_df)[1])
print("-----------------")
print("A1 fast mean", 100 * get_tracking_error(a1_fast_mean_df)[1])
print("A1 fast std", 100 * get_tracking_error(a1_fast_std_df)[1])
print("-----------------")
print("turn 05 mean", 100 * get_tracking_error(odds_turn_mean_df)[1])
print("turn 05 std", 100 * get_tracking_error(odds_turn_std_df)[1])
print("-----------------")
print("turn 1 mean", 100 * get_tracking_error(odds_turn_mid_mean_df)[1])
print("turn 1 std", 100 * get_tracking_error(odds_turn_mid_std_df)[1])
print("-----------------")
print("turn 15 mean", 100 * get_tracking_error(odds_turn_fast_mean_df)[1])
print("turn 15 std", 100 * get_tracking_error(odds_turn_fast_std_df)[1])
print("-----------------")

# +
fig, ax = plt.subplots(1,1, figsize=(12,6))
# plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_x], a1_fast_std_df[ulog_wrt_tar_x], label="Slow")
# plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_y], odds_turn_std_df[ulog_wrt_tar_y], label="wrt tar")
# plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_sp_wrt_tar_x], odds_turn_std_df[ulog_sp_wrt_tar_x], label="wrt tar")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_pos_error_norm], a1_slow_std_df[ulog_pos_error_norm], label="Slow")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_pos_error_norm], a1_slow_std_df[ulog_wrt_tar_pos_error_norm], label="Slow")

ax.legend()


# +
# Tracking Errors
fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_pos_error_norm], a1_slow_std_df[ulog_pos_error_norm], label="Slow")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_pos_error_norm], a1_fast_std_df[ulog_pos_error_norm], label="Fast")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('a1_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_vel_error_norm], a1_slow_std_df[ulog_vel_error_norm], label="Slow")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_vel_error_norm], a1_fast_std_df[ulog_vel_error_norm], label="Fast")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('a1_vel_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_pos_error_norm], odds_turn_std_df[ulog_pos_error_norm], label="0.5 m/s")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_pos_error_norm], odds_turn_mid_std_df[ulog_pos_error_norm], label="1 m/s")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_pos_error_norm], odds_turn_fast_std_df[ulog_pos_error_norm], label="1.5 m/s")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('turn_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_vel_error_norm], odds_turn_std_df[ulog_vel_error_norm], label="0.5 m/s")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_vel_error_norm], odds_turn_mid_std_df[ulog_vel_error_norm], label="1 m/s")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_vel_error_norm], odds_turn_fast_std_df[ulog_vel_error_norm], label="1.5 m/s")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('turn_vel_err.svg')

# +
# Target pose plots
fig, ax = plt.subplots(1,1, figsize=(4,4))
plot_mean_std(ax, a1_slow_mean_df[:4], a1_slow_mean_df[:4][mocap_target_y], a1_slow_std_df[:4][mocap_target_y], label="Slow")
plot_mean_std(ax, a1_fast_mean_df[:4], a1_fast_mean_df[:4][mocap_target_y], a1_fast_std_df[:4][mocap_target_y], label="Fast")
ax.axvline(x=[0], color='0', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(-5, 4.3)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Pos. [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('a1_pos.svg')

fig, ax = plt.subplots(1,1, figsize=(4,4))
plot_mean_std(ax, a1_slow_mean_df[:4], a1_slow_mean_df[:4][mocap_target_vy], a1_slow_std_df[:4][mocap_target_vy], label="Slow")
plot_mean_std(ax, a1_fast_mean_df[:4], a1_fast_mean_df[:4][mocap_target_vy], a1_fast_std_df[:4][mocap_target_vy], label="Fast")
ax.axvline(x=[0], color='0', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(-5, 4.3)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Vel. [m/s]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('a1_vel.svg')

fig, ax = plt.subplots(1,1, figsize=(4,4))
plot_mean_std(ax, odds_turn_mean_df[:4], odds_turn_mean_df[:4][mocap_target_x], odds_turn_std_df[:4][mocap_target_x], label="X")
plot_mean_std(ax, odds_turn_mean_df[:4], odds_turn_mean_df[:4][mocap_target_y], odds_turn_std_df[:4][mocap_target_y], label="Y")
ax.axvline(x=[0], color='0', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(-5, 4.3)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Pos. [m]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('turn_pos.svg')


fig, ax = plt.subplots(1,1, figsize=(4,4))
plot_mean_std(ax, odds_turn_mean_df[:4], odds_turn_mean_df[:4][mocap_target_vx], odds_turn_std_df[:4][mocap_target_vx], label="X")
plot_mean_std(ax, odds_turn_mean_df[:4], odds_turn_mean_df[:4][mocap_target_vy], odds_turn_std_df[:4][mocap_target_vy], label="Y")
ax.axvline(x=[0], color='0', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(-5, 4.3)
# ax.set_ylim(0, 0.25)
ax.set_ylabel('Vel. [m/s]')
ax.set_xlabel('Time [s]')
# ax.set_xticklabels([])
ax.legend()
fig.savefig('turn_vel.svg')


# +
# print(a1_fast_mean_df[mocap_target_vy][0:4].mean())
# print(odds_turn_mean_df[ulog_error_vx][4])
# print(a1_slow_mean_df[mocap_target_vy][4])
# fig, ax = plt.subplots(1,1)
# ax.plot(odds_turn_fast_mean_df[ulog_vy])
# ax.plot(odds_turn_fast_mean_df[mocap_target_vy])
# # ax.plot(odds_turn_fast_mean_df[ulog_vy] - odds_turn_fast_mean_df[mocap_target_vx])
# # odds_turn_fast_mean_df.plot(y=[ulog_vy])

for df in odds_turn_dfs:
    a = df[mocap_target_vx][0:4].values
    b = df[mocap_target_vy][0:4].values

    print(a)
    print(b)
    print(np.mean(np.linalg.norm((a,b), axis=0)))
    fig, ax = plt.subplots(1,1)
    ax.plot(np.linalg.norm((a,b), axis=0))
# print(np.linalg.norm(odds_turn_fast_mean_df[mocap_target_vx][0:4].values, odds_turn_fast_mean_df[mocap_target_vy][0:4].values))

# +
# Success Bar Plot
rates = {"a1_slow" : 10,
         "a1_fast" : 6,
         "turn_slow" : 10,
         "turn_mid" : 9,
         "turn_fast" : 7}


bars = (rates["a1_slow"], 
        rates["a1_fast"], 
        rates["turn_slow"],
        rates["turn_mid"],
        rates["turn_fast"])

fig, ax = plt.subplots(1,1, figsize=(12,6))

ind = np.arange(5)
width = 0.6

ax.bar(ind, bars, width)
ax.set_ylabel('Success')

ax.set_xticks(ind, ('Slow', 'Fast', '0.5 m/s', '1.0 m/s', '1.5 m/s'))
ax.legend(loc='best')
ax.axhline(y=10, color='0', ls='--', lw=2)
fig.savefig('moving_success.svg')

# -

# # Appendix

# +
# Longitudinal: + is in the forward direction (X)
# Lateral: if you are the drone facing the target, then left is positive (Y)
# Vertical: + points up (Z)

# For vision tests this means that:
# mavros_in_x is longitudinal, ulog_y is longitudinal
# mavros_in_y is lateral, ulog_x is lateral
# mavros_in_z is vertical, -ulog_z is vertical


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_error_y], medkit_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_error_x], medkit_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, medkit_mean_df, -medkit_mean_df[ulog_error_z], medkit_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Medkit 0.5 m/s")
ax.legend()
fig.savefig('appdx/medkit_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_error_y], pepsi_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_error_x], pepsi_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, pepsi_mean_df, -pepsi_mean_df[ulog_error_z], pepsi_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 0.5 m/s")
ax.legend()
fig.savefig('appdx/pepsi_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_error_y], cardboard_box_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_error_x], cardboard_box_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, cardboard_box_mean_df, -cardboard_box_mean_df[ulog_error_z], cardboard_box_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Cardboard Box 0.5 m/s")
ax.legend()
fig.savefig('appdx/cardboard_box_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_error_y], pepsi_mid_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_error_x], pepsi_mid_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, pepsi_mid_mean_df, -pepsi_mid_mean_df[ulog_error_z], pepsi_mid_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 1.25 m/s")
ax.legend()
fig.savefig('appdx/pepsi_125_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_y], pepsi_fast_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_x], pepsi_fast_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, pepsi_fast_mean_df, -pepsi_fast_mean_df[ulog_error_z], pepsi_fast_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 2 m/s")
ax.legend()
fig.savefig('appdx/pepsi_2_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_x], pepsi_fastest_std_df[mavros_error_x], label="Longitudinal")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_y], pepsi_fastest_std_df[mavros_error_y], label="Lateral")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_z], pepsi_fastest_std_df[mavros_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 3 m/s")
ax.legend()
fig.savefig('appdx/pepsi_3_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_x], outdoors_std_df[mavros_error_x], label="Longitudinal")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_y], outdoors_std_df[mavros_error_y], label="Lateral")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_z], outdoors_std_df[mavros_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Outdoors 0.5 m/s")
ax.legend()
fig.savefig('appdx/outdoors_05_pos_comps.svg')


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_error_vy], medkit_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_error_vx], medkit_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, medkit_mean_df, -medkit_mean_df[ulog_error_vz], medkit_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Medkit 0.5 m/s")
ax.legend()
fig.savefig('appdx/medkit_05_vel_comps.svg')
# fig.savefig('object_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_error_vy], pepsi_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_error_vx], pepsi_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, pepsi_mean_df, -pepsi_mean_df[ulog_error_vz], pepsi_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 0.5 m/s")
ax.legend()
fig.savefig('appdx/pepsi_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_error_vy], cardboard_box_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_error_vx], cardboard_box_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, cardboard_box_mean_df, -cardboard_box_mean_df[ulog_error_vz], cardboard_box_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Cardboard Box 0.5 m/s")
ax.legend()
fig.savefig('appdx/cardboard_box_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_error_vy], pepsi_mid_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_error_vx], pepsi_mid_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, pepsi_mid_mean_df, -pepsi_mid_mean_df[ulog_error_vz], pepsi_mid_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 1.25 m/s")
ax.legend()
fig.savefig('appdx/pepsi_125_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_vy], pepsi_fast_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_error_vx], pepsi_fast_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, pepsi_fast_mean_df, -pepsi_fast_mean_df[ulog_error_vz], pepsi_fast_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 2 m/s")
ax.legend()
fig.savefig('appdx/pepsi_2_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vx], pepsi_fastest_std_df[mavros_error_vx], label="Longitudinal")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vy], pepsi_fastest_std_df[mavros_error_vy], label="Lateral")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vz], pepsi_fastest_std_df[mavros_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 3 m/s")
ax.legend()
fig.savefig('appdx/pepsi_3_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vx], outdoors_std_df[mavros_error_vx], label="Longitudinal")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vy], outdoors_std_df[mavros_error_vy], label="Lateral")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vz], outdoors_std_df[mavros_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Outdoors 0.5 m/s")
ax.legend()
fig.savefig('appdx/outdoors_05_vel_comps.svg')

# +
# Longitudinal: + is in the forward direction (X)
# Lateral: if you are the drone facing the target, then left is positive (Y)
# Vertical: + points up (Z)

# For vision tests this means that:
# mavros_in_x is longitudinal, ulog_y is longitudinal
# mavros_in_y is lateral, ulog_x is lateral
# mavros_in_z is vertical, -ulog_z is vertical


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_x], medkit_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_y], medkit_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_z], medkit_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Medkit 0.5 m/s")
ax.legend()
fig.savefig('appdx/medkit_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_x], pepsi_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_y], pepsi_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_z], pepsi_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 0.5 m/s")
ax.legend()
fig.savefig('appdx/pepsi_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_x], cardboard_box_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_y], cardboard_box_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_z], cardboard_box_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Cardboard Box 0.5 m/s")
ax.legend()
fig.savefig('appdx/cardboard_box_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_x], pepsi_mid_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_y], pepsi_mid_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_z], pepsi_mid_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 1.25 m/s")
ax.legend()
fig.savefig('appdx/pepsi_125_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_x], pepsi_fast_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_y], pepsi_fast_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_z], pepsi_fast_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 2 m/s")
ax.legend()
fig.savefig('appdx/pepsi_2_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_wrt_tar_error_x], pepsi_fastest_std_df[mavros_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_wrt_tar_error_y], pepsi_fastest_std_df[mavros_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_wrt_tar_error_z], pepsi_fastest_std_df[mavros_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 3 m/s")
ax.legend()
fig.savefig('appdx/pepsi_3_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_wrt_tar_error_x], outdoors_std_df[mavros_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_wrt_tar_error_y], outdoors_std_df[mavros_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_wrt_tar_error_z], outdoors_std_df[mavros_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Outdoors 0.5 m/s")
ax.legend()
fig.savefig('appdx/outdoors_05_pos_comps.svg')


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_vx], medkit_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_vy], medkit_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, medkit_mean_df, medkit_mean_df[ulog_wrt_tar_error_vz], medkit_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Medkit 0.5 m/s")
ax.legend()
fig.savefig('appdx/medkit_05_vel_comps.svg')
# fig.savefig('object_pos_err.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_vx], pepsi_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_vy], pepsi_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, pepsi_mean_df, pepsi_mean_df[ulog_wrt_tar_error_vz], pepsi_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 0.5 m/s")
ax.legend()
fig.savefig('appdx/pepsi_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_vx], cardboard_box_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_vy], cardboard_box_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, cardboard_box_mean_df, cardboard_box_mean_df[ulog_wrt_tar_error_vz], cardboard_box_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Cardboard Box 0.5 m/s")
ax.legend()
fig.savefig('appdx/cardboard_box_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_vx], pepsi_mid_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_vy], pepsi_mid_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, pepsi_mid_mean_df, pepsi_mid_mean_df[ulog_wrt_tar_error_vz], pepsi_mid_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 1.25 m/s")
ax.legend()
fig.savefig('appdx/pepsi_125_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_vx], pepsi_fast_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_vy], pepsi_fast_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, pepsi_fast_mean_df, pepsi_fast_mean_df[ulog_wrt_tar_error_vz], pepsi_fast_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 2 m/s")
ax.legend()
fig.savefig('appdx/pepsi_2_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vx], pepsi_fastest_std_df[mavros_error_vx], label="Longitudinal")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vy], pepsi_fastest_std_df[mavros_error_vy], label="Lateral")
plot_mean_std(ax, pepsi_fastest_mean_df, pepsi_fastest_mean_df[mavros_error_vz], pepsi_fastest_std_df[mavros_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Two-liter 3 m/s")
ax.legend()
fig.savefig('appdx/pepsi_3_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vx], outdoors_std_df[mavros_error_vx], label="Longitudinal")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vy], outdoors_std_df[mavros_error_vy], label="Lateral")
plot_mean_std(ax, outdoors_mean_df, outdoors_mean_df[mavros_error_vz], outdoors_std_df[mavros_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Vision Outdoors 0.5 m/s")
ax.legend()
fig.savefig('appdx/outdoors_05_vel_comps.svg')

# +
# For A1:
# mocap_drone_y and ulog_x are longitudinal
# -mocap_drone_x and -ulog_y are lateral
# mocap_drone_z and -ulog_z are vertical

# For turntable:
# mocap_drone_x and ulog_y are longitudinal
# mocap_drone_y and ulog_x are lateral
# mocap_drone_z and -ulog_z are vertical

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_error_x], a1_slow_std_df[ulog_error_x], label="Longitudinal")
plot_mean_std(ax, a1_slow_mean_df, -a1_slow_mean_df[ulog_error_y], -a1_slow_std_df[ulog_error_y], label="Lateral")
plot_mean_std(ax, a1_slow_mean_df, -a1_slow_mean_df[ulog_error_z], a1_slow_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Slow")
ax.legend()
fig.savefig('appdx/a1_slow_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_error_x], a1_fast_std_df[ulog_error_x], label="Longitudinal")
plot_mean_std(ax, a1_fast_mean_df, -a1_fast_mean_df[ulog_error_y], -a1_fast_std_df[ulog_error_y], label="Lateral")
plot_mean_std(ax, a1_fast_mean_df, -a1_fast_mean_df[ulog_error_z], a1_fast_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Fast")
ax.legend()
fig.savefig('appdx/a1_fast_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_error_y], odds_turn_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_error_x], odds_turn_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, odds_turn_mean_df, -odds_turn_mean_df[ulog_error_z], odds_turn_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 0.5 m/s")
ax.legend()
fig.savefig('appdx/turn_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_error_y], odds_turn_mid_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_error_x], odds_turn_mid_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, odds_turn_mid_mean_df, -odds_turn_mid_mean_df[ulog_error_z], odds_turn_mid_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1 m/s")
ax.legend()
fig.savefig('appdx/turn_1_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_error_y], odds_turn_fast_std_df[ulog_error_y], label="Longitudinal")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_error_x], odds_turn_fast_std_df[ulog_error_x], label="Lateral")
plot_mean_std(ax, odds_turn_fast_mean_df, -odds_turn_fast_mean_df[ulog_error_z], odds_turn_fast_std_df[ulog_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1.5 m/s")
ax.legend()
fig.savefig('appdx/turn_15_pos_comps.svg')


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_error_vx], a1_slow_std_df[ulog_error_vx], label="Longitudinal")
plot_mean_std(ax, a1_slow_mean_df, -a1_slow_mean_df[ulog_error_vy], -a1_slow_std_df[ulog_error_vy], label="Lateral")
plot_mean_std(ax, a1_slow_mean_df, -a1_slow_mean_df[ulog_error_vz], a1_slow_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Slow")
ax.legend()
fig.savefig('appdx/a1_slow_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_error_vx], a1_fast_std_df[ulog_error_vx], label="Longitudinal")
plot_mean_std(ax, a1_fast_mean_df, -a1_fast_mean_df[ulog_error_vy], -a1_fast_std_df[ulog_error_vy], label="Lateral")
plot_mean_std(ax, a1_fast_mean_df, -a1_fast_mean_df[ulog_error_vz], a1_fast_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Fast")
ax.legend()
fig.savefig('appdx/a1_fast_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_error_vy], odds_turn_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_error_vx], odds_turn_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, odds_turn_mean_df, -odds_turn_mean_df[ulog_error_vz], odds_turn_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 0.5 m/s")
ax.legend()
fig.savefig('appdx/turn_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_error_vy], odds_turn_mid_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_error_vx], odds_turn_mid_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, odds_turn_mid_mean_df, -odds_turn_mid_mean_df[ulog_error_vz], odds_turn_mid_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1 m/s")
ax.legend()
fig.savefig('appdx/turn_1_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_error_vy], odds_turn_fast_std_df[ulog_error_vy], label="Longitudinal")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_error_vx], odds_turn_fast_std_df[ulog_error_vx], label="Lateral")
plot_mean_std(ax, odds_turn_fast_mean_df, -odds_turn_fast_mean_df[ulog_error_vz], odds_turn_fast_std_df[ulog_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1.5 m/s")
ax.legend()
fig.savefig('appdx/turn_15_vel_comps.svg')



# fig, ax = plt.subplots(1,1, figsize=(12,6))
# ax.plot(odds_turn_mean_df[ulog_x])
# ax.plot(odds_turn_mean_df[mocap_drone_y])

# +
# For A1:
# mocap_drone_y and ulog_x are longitudinal
# -mocap_drone_x and -ulog_y are lateral
# mocap_drone_z and -ulog_z are vertical

# For turntable:
# mocap_drone_x and ulog_y are longitudinal
# mocap_drone_y and ulog_x are lateral
# mocap_drone_z and -ulog_z are vertical

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_x], a1_slow_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_y], a1_slow_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_z], a1_slow_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Slow")
ax.legend()
fig.savefig('appdx/a1_slow_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_x], a1_fast_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_y], a1_fast_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_z], a1_fast_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Fast")
ax.legend()
fig.savefig('appdx/a1_fast_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_x], odds_turn_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_y], odds_turn_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_z], odds_turn_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 0.5 m/s")
ax.legend()
fig.savefig('appdx/turn_05_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_x], odds_turn_mid_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_y], odds_turn_mid_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_z], odds_turn_mid_std_df[ulog_wrt_tar_error_z], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1 m/s")
ax.legend()
fig.savefig('appdx/turn_1_pos_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_x], odds_turn_fast_std_df[ulog_wrt_tar_error_x], label="Longitudinal")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_y], odds_turn_fast_std_df[ulog_wrt_tar_error_y], label="Lateral")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_z], odds_turn_fast_std_df[ulog_wrt_tar_error_z], label="Vertical")

ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Pos. Errors [m]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1.5 m/s")
ax.legend()
fig.savefig('appdx/turn_15_pos_comps.svg')


fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_vx], a1_slow_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_vy], a1_slow_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, a1_slow_mean_df, a1_slow_mean_df[ulog_wrt_tar_error_vz], a1_slow_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Slow")
ax.legend()
fig.savefig('appdx/a1_slow_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_vx], a1_fast_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_vy], a1_fast_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, a1_fast_mean_df, a1_fast_mean_df[ulog_wrt_tar_error_vz], a1_fast_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Quadruped Fast")
ax.legend()
fig.savefig('appdx/a1_fast_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_vx], odds_turn_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_vy], odds_turn_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, odds_turn_mean_df, odds_turn_mean_df[ulog_wrt_tar_error_vz], odds_turn_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 0.5 m/s")
ax.legend()
fig.savefig('appdx/turn_05_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_vx], odds_turn_mid_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_vy], odds_turn_mid_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, odds_turn_mid_mean_df, odds_turn_mid_mean_df[ulog_wrt_tar_error_vz], odds_turn_mid_std_df[ulog_wrt_tar_error_vz], label="Vertical")
ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1 m/s")
ax.legend()
fig.savefig('appdx/turn_1_vel_comps.svg')

fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_vx], odds_turn_fast_std_df[ulog_wrt_tar_error_vx], label="Longitudinal")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_vy], odds_turn_fast_std_df[ulog_wrt_tar_error_vy], label="Lateral")
plot_mean_std(ax, odds_turn_fast_mean_df, odds_turn_fast_mean_df[ulog_wrt_tar_error_vz], odds_turn_fast_std_df[ulog_wrt_tar_error_vz], label="Vertical")

ax.axvline(x=[0], color='0.5', ls='--', lw=2)
ax.axvline(x=[4], color='0', ls='--', lw=2)
ax.set_xlim(0, 15)
ax.set_ylabel('Vel. Errors [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title("Mocap Turntable 1.5 m/s")
ax.legend()
fig.savefig('appdx/turn_15_vel_comps.svg')



# fig, ax = plt.subplots(1,1, figsize=(12,6))
# ax.plot(odds_turn_mean_df[ulog_x])
# ax.plot(odds_turn_mean_df[mocap_drone_y])

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


