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

sns.set()

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


def get_landing_bounds(dataframes, velocity, padding=1.0, num_devs=10.0):
    """Check for when landing happened."""
    acceleration = get_array_values(dfs["imu"], "accelerometer_m_s2", (3,))
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    mean, std = np.mean(acceleration_norm), np.std(acceleration_norm)

    land_threshold = mean + num_devs * std
    for i, norm in enumerate(acceleration_norm):
        if norm > land_threshold:
            return (
                dfs["imu"]["times"].iloc[i] - padding,
                dfs["imu"]["times"].iloc[i] + padding,
            )

    raise RuntimeError("Failed to find landing.")


# +
log_location = "../landing_experiments"
result_location = "../log_output"

padding = 1.0
num_devs = 10.0

result_dir = pathlib.Path(result_location).resolve()
log_files = get_files(log_location)

dfs = get_dataframes(result_dir, log_files[-8])
velocity = get_field_values(dfs["position"], ["vx", "vy", "vz"], (3,))
acceleration = get_array_values(dfs["imu"], "accelerometer_m_s2", (3,))
acceleration_norm = np.linalg.norm(acceleration, axis=1)

time_range = get_landing_bounds(dfs, velocity, padding=padding, num_devs=num_devs)


# +
fig, ax = plt.subplots(2, 1)

vel_norm = np.linalg.norm(velocity, axis=1)
vel_start = lookup_time_index(dfs["position"], time_range[0])
vel_end = lookup_time_index(dfs["position"], time_range[1])
vel_times = dfs["position"].iloc[vel_start:vel_end]["times"]

ax[0].plot(vel_times, vel_norm[vel_start:vel_end], label="norm", marker="+")
ax[0].plot(vel_times, velocity[vel_start:vel_end, 1], label="y", marker="+")
ax[0].plot(vel_times, velocity[vel_start:vel_end, 2], label="z", marker="+")
ax[0].axvline(time_range[0] + padding, label="landing start", ls="--", c="k")

ax[0].legend()
ax[0].set_title("Velocity")
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Velocity (m/s)")

acc_start = lookup_time_index(dfs["imu"], time_range[0])
acc_end = lookup_time_index(dfs["imu"], time_range[1])
acc_times = dfs["imu"].iloc[acc_start:acc_end]["times"]
ax[1].plot(acc_times, acceleration_norm[acc_start:acc_end], label="norm", marker="+")
ax[1].axvline(time_range[0] + padding, label="landing start", ls="--", c="k")

ax[1].legend()
ax[1].set_title("Acceleration Norm")
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Acceleration (m/s^2)")

fig.set_size_inches([15, 8])
plt.tight_layout()
plt.show()
# -


