# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
import bisect
import subprocess
import numpy as np
import functools
import pyquaternion as pyq
sns.set()

# +
MESSAGES = ["vehicle_local_position",
            "vehicle_local_position_setpoint",
            "position_setpoint_triplet",
            "vehicle_attitude",
            "vehicle_attitude_setpoint",
            "vehicle_rates_setpoint"]
MESSAGE_ARGS = ",".join(MESSAGES)

NAME_TO_MESSAGES = {
    "position": "vehicle_local_position_0",
    "position_sp": "vehicle_local_position_setpoint_0",
    "attitude": "vehicle_attitude_0",
    "attitude_sp": "vehicle_attitude_setpoint_0",
    "commanded": "position_setpoint_triplet_0",
    "attitude_output": "vehicle_rates_setpoint_0"
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
    return np.array([np.array([df.iloc[i][c] for c in columns]).reshape(shape) for i in range(len(df))])


def get_field_values(df, fields, shape):
    """Extract message fields into a numpy array."""
    total_size = functools.reduce(lambda x, y: x * y, shape, 1)
    return np.array([np.array([df.iloc[i][c] for c in fields]).reshape(shape) for i in range(len(df))])


def get_df_with_time(data_frames, name, first_time, timescale=1.0e-6):
    """Add a time column to the dataframe."""
    to_return = data_frames[name]
    to_return["times"] = (to_return["timestamp"] - first_time) * 1.0e-6
    return to_return


def get_dataframes(result_dir, file_to_use):
    """Get dataframes from the result directory."""
    message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))
    
    if len(message_csvs) == 0:
        subprocess.call(["ulog2csv", str(file_to_use), "-o", str(result_dir), "-m", MESSAGE_ARGS])
    message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))

    message_offset = len(file_to_use.stem) + 1
    data_frames = {filename.stem[message_offset:]: pd.read_csv(filename) for filename in message_csvs}
    first_timestamp = data_frames["vehicle_attitude_0"].iloc[0]["timestamp"]

    return {n: get_df_with_time(data_frames, m, first_timestamp) for n, m in NAME_TO_MESSAGES.items()}


def row_in_trajectory(row):
    """Check if a row belongs to a trajectory command."""
    return row["current.type"] == 0 and row["current.velocity_valid"]


def get_trajectory_bounds(dataframes, length, start_padding=3.0):
    """Get the trajectory time range if possible."""
    input_df = dataframes["commanded"]

    trajectory_times = []
    for i, time in enumerate(input_df["times"]):
        if row_in_trajectory(dataframes["commanded"].iloc[i]):
            trajectory_times.append(time)

    if len(trajectory_times) == 0:
        raise RuntimeError("Failed to find start time")
    
    
    start_time = min(trajectory_times) - start_padding
    return start_time, start_time + length + start_padding


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
    
    att_err = []
    pos_err = []
    vel_err = []
    
    for t in sample_times:
        t_actual = t + bounds[0]
        
        p = pos[lookup_time_index(dataframes["position"], t_actual), :]
        p_sp = pos_sp[lookup_time_index(dataframes["position_sp"], t_actual), :]
        pos_err.append(np.linalg.norm(p - p_sp))
        
        v = vel[lookup_time_index(dataframes["position"], t_actual), :]
        v_sp = vel_sp[lookup_time_index(dataframes["position_sp"], t_actual), :]
        vel_err.append(np.linalg.norm(v - v_sp))
        
        q = pyq.Quaternion(att[lookup_time_index(dataframes["attitude"], t_actual), :])
        q_sp = pyq.Quaternion(att_sp[lookup_time_index(dataframes["attitude_sp"], t_actual), :])
        q_err = q.inverse * q_sp
        att_err.append(abs(q_err.angle))
        
    return att_err, pos_err, vel_err

def plot_errors(ax, times, errors, label, **kwargs):
    mean = np.mean(errors, axis=1)
    stddev = np.std(errors, axis=1)
    handle = ax.plot(times, mean, label=label, **kwargs)[0]
    ax.fill_between(times, mean + stddev, mean - stddev, alpha=0.5, color=handle.get_color())


# +
adaptive = 0
geometric = 1
adaptive_log_location = "../adaptive_trials"
geometric_log_location = "../geometric_trials"
result_location = "../log_output"

start_padding = 1.0
length = 7.0
N_samples = 60
samples = np.linspace(0.0, length + start_padding, N_samples)

result_dir = pathlib.Path(result_location).resolve()

adaptive_files = get_files(adaptive_log_location)
geometric_files = get_files(geometric_log_location)

adap_att_errors = np.zeros((N_samples, len(adaptive_files)))
adap_pos_errors = np.zeros((N_samples, len(adaptive_files)))
adap_vel_errors = np.zeros((N_samples, len(adaptive_files)))

for i, filename in enumerate(adaptive_files):
    dfs = get_dataframes(result_dir, filename)
    time_range = get_trajectory_bounds(dfs, length, start_padding=start_padding)
    errors = get_errors(dfs, samples, time_range)
    
    adap_att_errors[:, i] = np.array(errors[0])
    adap_pos_errors[:, i] = np.array(errors[1])
    adap_vel_errors[:, i] = np.array(errors[2])
    
geom_att_errors = np.zeros((N_samples, len(geometric_files)))
geom_pos_errors = np.zeros((N_samples, len(geometric_files)))
geom_vel_errors = np.zeros((N_samples, len(geometric_files)))

for i, filename in enumerate(geometric_files):
    dfs = get_dataframes(result_dir, filename)
    time_range = get_trajectory_bounds(dfs, length, start_padding=start_padding)
    errors = get_errors(dfs, samples, time_range)
    
    geom_att_errors[:, i] = np.array(errors[0])
    geom_pos_errors[:, i] = np.array(errors[1])
    geom_vel_errors[:, i] = np.array(errors[2])    

# +
fig, ax = plt.subplots(3, 1)

plot_errors(ax[0], samples, adap_pos_errors, label="Adaptive")
plot_errors(ax[0], samples, geom_pos_errors, label="Geometric")
ax[0].legend()
ax[0].set_title("Adaptive vs Geometric (Position)")
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Error (meters)")

plot_errors(ax[1], samples, adap_vel_errors, label="Adaptive")
plot_errors(ax[1], samples, geom_vel_errors, label="Geometric")
ax[1].legend()
ax[1].set_title("Adaptive vs Geometric (Velocity)")
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Error (meters / second)")

plot_errors(ax[2], samples, adap_att_errors, label="adaptive")
plot_errors(ax[2], samples, geom_att_errors, label="Geometric")
ax[2].legend()
ax[2].set_title("Adaptive vs Geometric (Attitude)")
ax[2].set_xlabel("Time (seconds)")
ax[2].set_ylabel("Error (radians)")

fig.set_size_inches([15, 18])
plt.show()
# -


