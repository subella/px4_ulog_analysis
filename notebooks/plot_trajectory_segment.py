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

sns.set()

log_dir = pathlib.Path("../gripper_logs").resolve()
log_files = sorted(list(log_dir.glob("*.ulg")))
messages = [
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "position_setpoint_triplet",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
]
message_args = ",".join(messages)

show_all = False
file_to_use = log_files[-1]
padding = 1.5
print("Opening {}".format(file_to_use))

# +
result_dir = pathlib.Path("../log_output").resolve()
message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))
if len(message_csvs) == 0:
    subprocess.call(
        ["ulog2csv", str(file_to_use), "-o", str(result_dir), "-m", message_args]
    )
message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))

message_offset = len(file_to_use.stem) + 1
data_frames = {
    filename.stem[message_offset:]: pd.read_csv(filename) for filename in message_csvs
}
first_timestamp = data_frames["vehicle_attitude_0"].iloc[0]["timestamp"]


def get_df(data_frames, name, first_time, timescale=1.0e-6):
    to_return = data_frames[name]
    to_return["times"] = (to_return["timestamp"] - first_timestamp) * 1.0e-6
    return to_return


def plot_sp(
    ax, actual_df, desired_df, actual_labels, desired_labels, start=0.0, end=1000.0
):
    act_start = bisect.bisect_left(actual_df["times"], start) - 1
    act_end = bisect.bisect_left(actual_df["times"], end)
    des_start = bisect.bisect_left(desired_df["times"], start) - 1
    des_end = bisect.bisect_left(desired_df["times"], end) - 1
    actual_df = actual_df.iloc[act_start:act_end]
    desired_df = desired_df.iloc[des_start:des_end]

    colors = sns.color_palette()

    for i, l in enumerate(actual_labels):
        ax.plot(actual_df["times"], actual_df[l], label=l, c=colors[i])
    for i, l in enumerate(desired_labels):
        ax.plot(
            desired_df["times"],
            desired_df[l],
            label="{} (setpoint)".format(l),
            c=colors[i],
            ls="--",
        )


# +
input_df = get_df(data_frames, "position_setpoint_triplet_0", first_timestamp)

trajectory_times = []
for i, time in enumerate(input_df["times"]):
    if (
        input_df.iloc[i]["current.type"] == 0
        and input_df.iloc[i]["current.velocity_valid"]
    ):
        trajectory_times.append(time)

if len(trajectory_times) > 0 and not show_all:
    start_time = min(trajectory_times) - padding
    end_time = max(trajectory_times) + padding
else:
    start_time = input_df.iloc[0]["times"]
    end_time = input_df.iloc[-1]["times"]

local_pose_df = get_df(data_frames, "vehicle_local_position_0", first_timestamp)
attitude_df = get_df(data_frames, "vehicle_attitude_0", first_timestamp)
local_pose_sp_df = get_df(
    data_frames, "vehicle_local_position_setpoint_0", first_timestamp
)
attitude_sp_df = get_df(data_frames, "vehicle_attitude_setpoint_0", first_timestamp)

# +
fig, ax = plt.subplots(1, 2)

plot_sp(
    ax[0],
    local_pose_df,
    local_pose_sp_df,
    ["x", "y", "z"],
    ["x", "y", "z"],
    start=start_time,
    end=end_time,
)
ax[0].set_title("Position Control")
ax[0].legend()
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Position (meters)")

plot_sp(
    ax[1],
    local_pose_df,
    local_pose_sp_df,
    ["vx", "vy", "vz"],
    ["vx", "vy", "vz"],
    start=start_time,
    end=end_time,
)
ax[1].set_title("Velocity Control")
ax[1].legend()
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Velocity (meters/second)")

fig.set_size_inches([15, 6])
plt.show()

# +
fig, ax = plt.subplots(1, 2)

plot_sp(
    ax[0],
    attitude_df,
    attitude_sp_df,
    ["q[0]", "q[1]", "q[2]", "q[3]"],
    ["q_d[0]", "q_d[1]", "q_d[2]", "q_d[3]"],
    start=start_time,
    end=end_time,
)
ax[0].set_title("Attitude Control")
ax[0].legend()
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Attitude (radians)")

plot_sp(
    ax[1],
    attitude_df,
    attitude_sp_df,
    ["rollspeed", "pitchspeed", "yawspeed"],
    ["angle_vel_x", "angle_vel_y", "angle_vel_z"],
    start=start_time,
    end=end_time,
)
ax[1].set_title("Angular Velocity Control")
ax[1].legend()
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Angular Velocity (radians/second)")

fig.set_size_inches([15, 6])
plt.show()
