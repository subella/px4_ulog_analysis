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


# +
MESSAGES = [
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "sensor_combined",
]
MESSAGE_ARGS = ",".join(MESSAGES)

NAME_TO_MESSAGES = {
    "position": "vehicle_local_position_0",
    "position_sp": "vehicle_local_position_setpoint_0",
    "imu": "sensor_combined_0",
}
TIME_AUTHORITY_MESSAGE = "sensor_combined_0"


# +
def get_files(log_location):
    """Get the file from the index."""
    log_dir = pathlib.Path(log_location).resolve()
    return sorted(list(log_dir.glob("*.ulg")))


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
    subprocess.call(
        ["ulog2csv", str(file_to_use), "-o", str(result_dir), "-m", MESSAGE_ARGS]
    )
    message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))

    message_offset = len(file_to_use.stem) + 1
    data_frames = {
        filename.stem[message_offset:]: pd.read_csv(filename)
        for filename in message_csvs
    }
    first_timestamp = data_frames[TIME_AUTHORITY_MESSAGE].iloc[0]["timestamp"]

    return {
        n: get_df_with_time(data_frames, m, first_timestamp)
        for n, m in NAME_TO_MESSAGES.items()
    }


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


def get_landing_bounds(dataframes, accelerations, velocities, stop_threshold=0.3):
    """Check for when landing happened."""
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    max_idx = np.argmax(acceleration_norm)

    start = dataframes["imu"]["times"].iloc[max_idx]
    end = dataframes["imu"]["times"].iloc[-1]

    vel_start_idx = lookup_time_index(dataframes["position"], start)
    for i in range(vel_start_idx, velocities.shape[0]):
        if np.linalg.norm(velocities[i, :]) < stop_threshold:
            end = dataframes["position"]["times"].iloc[i]
            break

    return start, end


def plot_landing_velocity(ax, dataframes, velocities, bounds, padding=1.0):
    """Show the velocity of the drone during landing."""
    vel_norm = np.linalg.norm(velocities, axis=1)
    vel_start = lookup_time_index(dfs["position"], bounds[0] - padding)
    vel_end = lookup_time_index(dfs["position"], bounds[1] + padding)
    vel_times = dfs["position"].iloc[vel_start:vel_end]["times"] - (bounds[0] - padding)

    ax.plot(vel_times, vel_norm[vel_start:vel_end], label="norm", marker="+")
    ax.plot(vel_times, velocities[vel_start:vel_end, 1], label="y", marker="+")
    ax.plot(vel_times, velocities[vel_start:vel_end, 2], label="z", marker="+")
    ax.axvline(padding, label="landing start", ls="--", c="k")
    ax.axvline(bounds[1] - bounds[0] + padding, label="landing end", ls="--", c="k")

    ax.legend()
    ax.set_title("Landing Velocity")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Velocity (m/s)")


def plot_landing_acceleration(ax, dataframes, accelerations, bounds, padding=1.0):
    """Show the acceleration of the drone during landing."""
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    acc_start = lookup_time_index(dfs["imu"], bounds[0] - padding)
    acc_end = lookup_time_index(dfs["imu"], bounds[1] + padding)
    acc_times = dfs["imu"].iloc[acc_start:acc_end]["times"] - (bounds[0] - padding)
    ax.plot(acc_times, acceleration_norm[acc_start:acc_end], label="norm", marker="+")
    ax.axvline(padding, label="landing start", ls="--", c="k")
    ax.axvline(bounds[1] - bounds[0] + padding, label="landing end", ls="--", c="k")

    ax.legend()
    ax.set_title("Acceleration Norm")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration (m/s^2)")


def create_trial_summary(dataframes, stop_threshold=0.3, padding=1.0):
    """Make a numpy array out of the trial."""
    velocities = get_field_values(dataframes["position"], ["vx", "vy", "vz"], (3,))
    accelerations = get_array_values(dataframes["imu"], "accelerometer_m_s2", (3,))

    time_range = get_landing_bounds(
        dataframes, accelerations, velocities, stop_threshold=stop_threshold
    )

    vel_norm = np.linalg.norm(velocities, axis=1)
    vel_start = lookup_time_index(dataframes["position"], time_range[0] - padding)
    vel_end = lookup_time_index(dataframes["position"], time_range[1] + padding)
    vel_times = dataframes["position"].iloc[vel_start:vel_end]["times"] - (
        time_range[0] - padding
    )

    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    acc_start = lookup_time_index(dataframes["imu"], time_range[0] - padding)
    acc_end = lookup_time_index(dataframes["imu"], time_range[1] + padding)
    acc_times = dataframes["imu"].iloc[acc_start:acc_end]["times"] - (
        time_range[0] - padding
    )

    vel_results = np.zeros((vel_end - vel_start, 4))
    vel_results[:, 0] = vel_times
    vel_results[:, 1] = velocities[vel_start:vel_end, 1]
    vel_results[:, 2] = velocities[vel_start:vel_end, 2]
    vel_results[:, 3] = vel_norm[vel_start:vel_end]

    acc_results = np.zeros((acc_end - acc_start, 2))
    acc_results[:, 0] = acc_times
    acc_results[:, 1] = acceleration_norm[acc_start:acc_end]

    return (padding, time_range[1] - time_range[0] + padding), vel_results, acc_results


# +
naive_log_location = "../naive_landing_trials"
optimal_log_location = "../optimal_landing_trials"
result_location = "../log_output"

padding = 0.4
stop_threshold = 0.3
index = -1

result_dir = pathlib.Path(result_location).resolve()
naive_files = get_files(naive_log_location)
optimal_files = get_files(optimal_log_location)

naive_dfs = [get_dataframes(result_dir, filename) for filename in naive_files]
optimal_dfs = [get_dataframes(result_dir, filename) for filename in optimal_files]

naive_results = [
    create_trial_summary(dfs, stop_threshold=stop_threshold, padding=padding) for dfs in naive_dfs
]
optimal_results = [
    create_trial_summary(dfs, stop_threshold=stop_threshold, padding=padding) for dfs in optimal_dfs
]


# +

sns.set()
sns.set_style("whitegrid")

def plot_trials(ax, results, kind, color, dim):
    """Plot some info."""
    titles = ["Y Velocity", "Z Velocity", "Velocity Norm", "Acceleration Norm"]
    y_labels = ["Velocity (m/s)", "Velocity (m/s)", "Velocity (m/s)", "Acceleration (m/s^2)"]

    bounds = results[0][0]
    ax.axvline(bounds[0], label="landing start", ls="--", c="k", alpha=0.4)

    result_index = 1 if dim < 3 else 2
    offset = 1 if dim < 3 else -2
    for result in results:
        ax.plot(result[result_index][:, 0], result[result_index][:, dim + offset], c=color, alpha=0.8, label=kind)

    ax.set_title(titles[dim])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_labels[dim])
    ax.legend()


colors = sns.color_palette()

fig, ax = plt.subplots(4, 1, sharex=True)

for i in range(4):
    plot_trials(ax[i], naive_results, "naive", colors[0], i)
    plot_trials(ax[i], optimal_results, "optimal", colors[1], i)

fig.set_size_inches([10, 20])
plt.tight_layout()
plt.show()
# -

