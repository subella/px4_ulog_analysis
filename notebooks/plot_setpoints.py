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

sns.set()

log_dir = pathlib.Path("../logs").resolve()
log_files = sorted(list(log_dir.glob("*.ulg")))
print("Available files:")
for log_file in log_files:
    print("- {}".format(log_file))

file_to_use = log_files[-2]
start_time = 17
end_time = 40
show_all = False

result_dir = pathlib.Path("../log_output").resolve()
message_csvs = list(result_dir.glob("{}*".format(file_to_use.stem)))
message_offset = len(file_to_use.stem) + 1
data_frames = {
    filename.stem[message_offset:]: pd.read_csv(filename) for filename in message_csvs
}
first_timestamp = data_frames["input_rc_0"].iloc[0]["timestamp"]

# +
input_df = data_frames["input_rc_0"]
input_df["times"] = (input_df["timestamp"] - first_timestamp) * 1.0e-6

plt.plot(input_df["times"], input_df["values[6]"], ls="none", marker="o")
plt.show()

killtimes = []
for index, value in enumerate(input_df["values[6]"].diff()):
    if value > 100:
        killtimes.append(input_df["times"][index])
killtimes = [killtimes[-1]]
# -

local_pose_df = data_frames["vehicle_local_position_0"]
local_pose_df["times"] = (local_pose_df["timestamp"] - first_timestamp) * 1.0e-6
local_pose_df.head()

attitude_df = data_frames["vehicle_attitude_0"]
attitude_df["times"] = (attitude_df["timestamp"] - first_timestamp) * 1.0e-6
attitude_df.head()

local_pose_sp_df = data_frames["vehicle_local_position_setpoint_0"]
local_pose_sp_df["times"] = (local_pose_sp_df["timestamp"] - first_timestamp) * 1.0e-6
local_pose_sp_df.head()

attitude_sp_df = data_frames["vehicle_attitude_setpoint_0"]
attitude_sp_df["times"] = (attitude_sp_df["timestamp"] - first_timestamp) * 1.0e-6
attitude_sp_df.head()

actuator_df = data_frames["actuator_controls_0_0"]
actuator_df["times"] = (actuator_df["timestamp"] - first_timestamp) * 1.0e-6
actuator_df.head()

# +
if show_all:
    actual_df, desired_df = local_pose_df, local_pose_sp_df
else:
    act_start = bisect.bisect_left(local_pose_df["times"], start_time) - 1
    act_end = bisect.bisect_left(local_pose_df["times"], end_time)
    des_start = bisect.bisect_left(local_pose_sp_df["times"], start_time) - 1
    des_end = bisect.bisect_left(local_pose_sp_df["times"], end_time) - 1
    actual_df = local_pose_df.iloc[act_start:act_end]
    desired_df = local_pose_sp_df.iloc[des_start:des_end]

colors = sns.color_palette()

plt.plot(actual_df["times"], actual_df["x"], label="x", c=colors[0])
plt.plot(actual_df["times"], actual_df["y"], label="y", c=colors[1])
plt.plot(actual_df["times"], actual_df["z"], label="z", c=colors[2])
plt.plot(
    desired_df["times"], desired_df["x"], label="x (setpoint)", c=colors[0], ls="--"
)
plt.plot(
    desired_df["times"], desired_df["y"], label="y (setpoint)", c=colors[1], ls="--"
)
plt.plot(
    desired_df["times"], desired_df["z"], label="z (setpoint)", c=colors[2], ls="--"
)

for killtime in killtimes:
    if killtime < end_time:
        plt.gca().axvline(killtime, c="k")

plt.title("Position Control")
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.gcf().set_size_inches([12, 6])
plt.show()

# +
if show_all:
    actual_df, desired_df = attitude_df, attitude_sp_df
else:
    act_start = bisect.bisect_left(attitude_df["times"], start_time) - 1
    act_end = bisect.bisect_left(attitude_df["times"], end_time)
    des_start = bisect.bisect_left(attitude_sp_df["times"], start_time) - 1
    des_end = bisect.bisect_left(attitude_sp_df["times"], end_time) - 1
    actual_df = attitude_df.iloc[act_start:act_end]
    desired_df = attitude_sp_df.iloc[des_start:des_end]

colors = sns.color_palette()

plt.plot(actual_df["times"], actual_df["q[0]"], label="qw", c=colors[0])
plt.plot(actual_df["times"], actual_df["q[1]"], label="qx", c=colors[1])
plt.plot(actual_df["times"], actual_df["q[2]"], label="qy", c=colors[2])
plt.plot(actual_df["times"], actual_df["q[3]"], label="qz", c=colors[3])
plt.plot(
    desired_df["times"],
    desired_df["q_d[0]"],
    label="qw (setpoint)",
    c=colors[0],
    ls="--",
)
plt.plot(
    desired_df["times"],
    desired_df["q_d[1]"],
    label="qx (setpoint)",
    c=colors[1],
    ls="--",
)
plt.plot(
    desired_df["times"],
    desired_df["q_d[2]"],
    label="qy (setpoint)",
    c=colors[2],
    ls="--",
)
plt.plot(
    desired_df["times"],
    desired_df["q_d[3]"],
    label="qz (setpoint)",
    c=colors[3],
    ls="--",
)

for killtime in killtimes:
    if killtime < end_time:
        plt.gca().axvline(killtime, c="k")

plt.title("Attitude Control")
plt.ylabel("Quaternion Component (radians)")
plt.xlabel("Time (seconds)")
plt.legend()
plt.gcf().set_size_inches([12, 6])
plt.show()

# +
if show_all:
    actual_df, desired_df = local_pose_df, local_pose_sp_df
else:
    act_start = bisect.bisect_left(local_pose_df["times"], start_time) - 1
    act_end = bisect.bisect_left(local_pose_df["times"], end_time)
    des_start = bisect.bisect_left(local_pose_sp_df["times"], start_time) - 1
    des_end = bisect.bisect_left(local_pose_sp_df["times"], end_time) - 1
    actual_df = local_pose_df.iloc[act_start:act_end]
    desired_df = local_pose_sp_df.iloc[des_start:des_end]

colors = sns.color_palette()

plt.plot(actual_df["times"], actual_df["vx"], label="vx", c=colors[0])
plt.plot(actual_df["times"], actual_df["vy"], label="vy", c=colors[1])
plt.plot(actual_df["times"], actual_df["vz"], label="vz", c=colors[2])
plt.plot(
    desired_df["times"], desired_df["vx"], label="vx (setpoint)", c=colors[0], ls="--"
)
plt.plot(
    desired_df["times"], desired_df["vy"], label="vy (setpoint)", c=colors[1], ls="--"
)
plt.plot(
    desired_df["times"], desired_df["vz"], label="vz (setpoint)", c=colors[2], ls="--"
)

for killtime in killtimes:
    if killtime < end_time:
        plt.gca().axvline(killtime, c="k")

plt.title("Velocity Control")
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (meters/s)")
plt.gcf().set_size_inches([12, 6])
plt.show()

# +
if show_all:
    actual_df, desired_df = attitude_df, attitude_sp_df
else:
    act_start = bisect.bisect_left(attitude_df["times"], start_time) - 1
    act_end = bisect.bisect_left(attitude_df["times"], end_time)
    des_start = bisect.bisect_left(attitude_sp_df["times"], start_time) - 1
    des_end = bisect.bisect_left(attitude_sp_df["times"], end_time) - 1
    actual_df = attitude_df.iloc[act_start:act_end]
    desired_df = attitude_sp_df.iloc[des_start:des_end]

colors = sns.color_palette()
plt.plot(actual_df["times"], actual_df["rollspeed"], label="x", c=colors[0])
plt.plot(actual_df["times"], actual_df["pitchspeed"], label="y", c=colors[1])
plt.plot(actual_df["times"], actual_df["yawspeed"], label="z", c=colors[2])
plt.plot(
    desired_df["times"],
    desired_df["angle_vel_x"],
    label="x (setpoint)",
    c=colors[0],
    ls="--",
)
plt.plot(
    desired_df["times"],
    desired_df["angle_vel_y"],
    label="y (setpoint)",
    c=colors[1],
    ls="--",
)
plt.plot(
    desired_df["times"],
    desired_df["angle_vel_z"],
    label="z (setpoint)",
    c=colors[2],
    ls="--",
)

for killtime in killtimes:
    if killtime < end_time:
        plt.gca().axvline(killtime, c="k")

plt.title("Angular Velocity Control")
plt.ylabel("Angular Velocity (radians/second)")
plt.xlabel("Time (seconds)")
plt.legend()
plt.gcf().set_size_inches([12, 6])
plt.show()

# +
if show_all:
    actual_df = actuator_df
else:
    act_start = bisect.bisect_left(actuator_df["times"], start_time) - 1
    act_end = bisect.bisect_left(actuator_df["times"], end_time)
    actual_df = actuator_df.iloc[act_start:act_end]

colors = sns.color_palette()
plt.plot(actual_df["times"], actual_df["control[0]"], label="roll", c=colors[0])
plt.plot(actual_df["times"], actual_df["control[1]"], label="pitch", c=colors[1])
plt.plot(actual_df["times"], actual_df["control[2]"], label="yaw", c=colors[2])
plt.plot(actual_df["times"], actual_df["control[3]"], label="thrust", c=colors[3])

for killtime in killtimes:
    if killtime < end_time:
        plt.gca().axvline(killtime, c="k")

plt.title("Actuator Output")
plt.xlabel("Time (seconcds)")
plt.ylabel("Actuator Command")
plt.legend()
plt.gcf().set_size_inches([12, 6])
plt.show()
