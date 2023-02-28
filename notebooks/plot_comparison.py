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
# def get_filename(log_dir, index):
#     """Get the file from the index."""
#     return next(log_dir.glob("{:05d}*.ulg".format(index)))

def get_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return list(log_dir.glob("*.ulg"))[-1]


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
    return row["current.type"] == 0 and row["current.velocity_valid"]


def get_trajectory_bounds(
    dataframes, show_all=False, start_padding=3.0, end_padding=1.0
):
    """Get the trajectory time range if possible."""
    input_df = dataframes["commanded"]

    trajectory_times = []
    for i, time in enumerate(input_df["times"]):
        if row_in_trajectory(dataframes["commanded"].iloc[i]):
            trajectory_times.append(time)

    if len(trajectory_times) > 0 and not show_all:
        start_time = min(trajectory_times) - start_padding
        end_time = max(trajectory_times) + end_padding
    else:
        start_time = dataframes["attitude"].iloc[0]["times"]
        end_time = dataframes["attitude"].iloc[-1]["times"]

    return start_time, end_time


def lookup_time_index(df, t):
    """Get the index nearest to the time."""
    idx = bisect.bisect_right(df["times"], t)

    if idx == 0:
        return idx

    diff = df["times"][idx] - t
    other_diff = df["times"][idx - 1] - t

    if other_diff < diff:
        return idx - 1


def get_pos_errors(dataframes, bounds):
    """Get errors for specific run."""
    pos = get_field_values(dataframes["position"], ["x", "y", "z"], (3,))
    vel = get_field_values(dataframes["position"], ["vx", "vy", "vz"], (3,))
    pos_sp = get_field_values(dataframes["position_sp"], ["x", "y", "z"], (3,))
    vel_sp = get_field_values(dataframes["position_sp"], ["vx", "vy", "vz"], (3,))

    times = []
    pos_err = []
    vel_err = []
    pos_start_idx = lookup_time_index(dataframes["position_sp"], bounds[0])
    pos_end_idx = lookup_time_index(dataframes["position_sp"], bounds[1])
    for i in range(pos_start_idx, pos_end_idx):
        times.append(dataframes["position_sp"]["times"][i] - bounds[0])
        to_compare = lookup_time_index(
            dataframes["position"], dataframes["position_sp"]["times"][i]
        )
        pos_err.append(np.linalg.norm(pos[to_compare, :] - pos_sp[i, :]))
        vel_err.append(np.linalg.norm(vel[to_compare, :] - vel_sp[i, :]))
    return times, pos_err, vel_err


def get_att_errors(dataframes, bounds):
    """Get attitude errors for a specific run."""
    att = get_array_values(dataframes["attitude"], "q", (4,))
    att_sp = get_array_values(dataframes["attitude_sp"], "q_d", (4,))

    times = []
    att_err = []
    att_start_idx = lookup_time_index(dataframes["attitude_sp"], bounds[0])
    att_end_idx = lookup_time_index(dataframes["attitude_sp"], bounds[1])
    for i in range(att_start_idx, att_end_idx):
        times.append(dataframes["attitude_sp"]["times"][i] - bounds[0])
        to_compare = lookup_time_index(
            dataframes["attitude"], dataframes["attitude_sp"]["times"][i]
        )
        q = pyq.Quaternion(att[to_compare, :])
        q_sp = pyq.Quaternion(att_sp[i, :])
        q_err = q.inverse * q_sp
        att_err.append(abs(q_err.angle))

    return times, att_err


def plot_adaptive(ax, dataframes, name, fields, bounds, postfix="", **kwargs):
    """Plot adaptive terms."""
    deltas = get_field_values(dataframes[name], fields, (3,))
    start_idx = lookup_time_index(dataframes[name], bounds[0])
    end_idx = lookup_time_index(dataframes[name], bounds[1])

    colors = sns.color_palette()
    labels = ["{}{}".format(l, postfix) for l in ("x", "y", "z")]
    ax.plot(
        dataframes[name].iloc[start_idx:end_idx]["times"] - bounds[0],
        deltas[start_idx:end_idx, 0],
        label=labels[0],
        c=colors[0],
        **kwargs
    )
    ax.plot(
        dataframes[name].iloc[start_idx:end_idx]["times"] - bounds[0],
        deltas[start_idx:end_idx, 1],
        label=labels[1],
        c=colors[1],
        **kwargs
    )
    ax.plot(
        dataframes[name].iloc[start_idx:end_idx]["times"] - bounds[0],
        deltas[start_idx:end_idx, 2],
        label=labels[2],
        c=colors[2],
        **kwargs
    )


# +
adaptive = 0
geometric = 1
log_location = "../logs"
result_location = "../log_output"
show_geometric_adaptive_terms = False

start_padding = 1.0
end_padding = 4.0

log_dir = pathlib.Path(log_location).resolve()
result_dir = pathlib.Path(result_location).resolve()
adaptive_file = get_files(log_dir)
# geometric_file = get_filename(log_dir, geometric)

print(adaptive_file)
# print(geometric_file)

adaptive_dfs = get_dataframes(result_dir, adaptive_file)
# geometric_dfs = get_dataframes(result_dir, geometric_file)
adaptive_range = get_trajectory_bounds(
    adaptive_dfs, start_padding=start_padding, end_padding=end_padding
)
# geometric_range = get_trajectory_bounds(
#     geometric_dfs, start_padding=start_padding, end_padding=end_padding
# )

adaptive_pos_errors = get_pos_errors(adaptive_dfs, adaptive_range)
# geometric_pos_errors = get_pos_errors(geometric_dfs, geometric_range)
adaptive_att_errors = get_att_errors(adaptive_dfs, adaptive_range)
# geometric_att_errors = get_att_errors(geometric_dfs, geometric_range)

# +
fig, ax = plt.subplots(3, 2)

# Errors

ax[0][0].plot(adaptive_pos_errors[0], adaptive_pos_errors[1], label="adaptive")
ax[0][0].plot(geometric_pos_errors[0], geometric_pos_errors[1], label="geometric")
ax[0][0].legend()
ax[0][0].set_title("Adaptive vs Geometric (Position)")
ax[0][0].set_xlabel("Time (seconds)")
ax[0][0].set_ylabel("Error (meters)")

ax[1][0].plot(adaptive_pos_errors[0], adaptive_pos_errors[2], label="adaptive")
ax[1][0].plot(geometric_pos_errors[0], geometric_pos_errors[2], label="geometric")
ax[1][0].legend()
ax[1][0].set_title("Adaptive vs Geometric (Velocity)")
ax[1][0].set_xlabel("Time (seconds)")
ax[1][0].set_ylabel("Error (meters / second)")

ax[2][0].plot(adaptive_att_errors[0], adaptive_att_errors[1], label="adaptive")
ax[2][0].plot(geometric_att_errors[0], geometric_att_errors[1], label="geometric")
ax[2][0].legend()
ax[2][0].set_title("Adaptive vs Geometric (Attitude)")
ax[2][0].set_xlabel("Time (seconds)")
ax[2][0].set_ylabel("Error (radians)")

# Adaptive terms
plot_adaptive(
    ax[0][1],
    adaptive_dfs,
    "attitude_sp",
    ["adap_x", "adap_y", "adap_z"],
    adaptive_range,
)
if show_geometric_adaptive_terms:
    plot_adaptive(
        ax[0][1],
        geometric_dfs,
        "attitude_sp",
        ["adap_x", "adap_y", "adap_z"],
        geometric_range,
        postfix=" (geometric)",
        ls="--",
    )

ax[0][1].set_title("Adaptive Term (Position)")
ax[0][1].legend()
ax[0][1].set_xlabel("Time (seconds)")
ax[0][1].set_ylabel("Force (kg * m/s^2)")

plot_adaptive(
    ax[1][1],
    adaptive_dfs,
    "attitude_sp",
    ["adap_vx", "adap_vy", "adap_vz"],
    adaptive_range,
)
if show_geometric_adaptive_terms:
    plot_adaptive(
        ax[1][1],
        geometric_dfs,
        "attitude_sp",
        ["adap_vx", "adap_vy", "adap_vz"],
        geometric_range,
        postfix=" (geometric)",
        ls="--",
    )

ax[1][1].set_title("Adaptive Term (Position Derivative)")
ax[1][1].legend()
ax[1][1].set_xlabel("Time (seconds)")
ax[1][1].set_ylabel("Jerk (kg * m/s^3)")

plot_adaptive(
    ax[2][1],
    adaptive_dfs,
    "attitude_output",
    ["adap_x", "adap_y", "adap_z"],
    adaptive_range,
)
if show_geometric_adaptive_terms:
    plot_adaptive(
        ax[2][1],
        geometric_dfs,
        "attitude_output",
        ["adap_x", "adap_y", "adap_z"],
        geometric_range,
        postfix=" (geometric)",
        ls="--",
    )

ax[2][1].set_title("Adaptive Term (Attitude)")
ax[2][1].legend()
ax[2][1].set_xlabel("Time (seconds)")
ax[2][1].set_ylabel("Moment (kg * rad/s^2)")

fig.set_size_inches([15, 18])
plt.show()
# -


