#!/usr/bin/env python3
"""
Dump IMU and TDOA raw data to CSV files for offline debugging.

Usage:
  1. Copy this script to your workspace
  2. chmod +x dump_raw_data.py
  3. rosrun uwb_imu_fusion dump_raw_data.py
     OR: python3 dump_raw_data.py
  4. Play your rosbag or run live for 30-60 seconds
  5. Ctrl+C to stop — files are written to /tmp/

Outputs:
  /tmp/imu_raw.csv       - timestamp, ax, ay, az, gx, gy, gz
  /tmp/tdoa_raw.csv      - timestamp, idA, idB, tdoa_measurement
  /tmp/data_summary.txt  - quick statistics for debugging
"""

import rospy
import csv
import signal
import sys
import numpy as np
from sensor_msgs.msg import Imu
from cf_msgs.msg import Tdoa

imu_data = []
tdoa_data = []

def imu_cb(msg):
    imu_data.append([
        msg.header.stamp.to_sec(),
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    ])

def tdoa_cb(msg):
    tdoa_data.append([
        msg.header.stamp.to_sec(),
        msg.idA,
        msg.idB,
        msg.data,
    ])

def save_and_exit(*args):
    rospy.loginfo("Saving %d IMU and %d TDOA samples...", len(imu_data), len(tdoa_data))

    # --- IMU CSV ---
    with open("/tmp/imu_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        w.writerows(imu_data)

    # --- TDOA CSV ---
    with open("/tmp/tdoa_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "idA", "idB", "tdoa"])
        w.writerows(tdoa_data)

    # --- Summary ---
    with open("/tmp/data_summary.txt", "w") as f:
        f.write("=== DATA DUMP SUMMARY ===\n\n")

        if imu_data:
            imu = np.array(imu_data)
            dt_imu = np.diff(imu[:, 0])
            f.write(f"IMU samples: {len(imu_data)}\n")
            f.write(f"IMU duration: {imu[-1, 0] - imu[0, 0]:.3f} s\n")
            f.write(f"IMU rate: {1.0 / np.mean(dt_imu):.1f} Hz (mean dt={np.mean(dt_imu)*1000:.2f} ms)\n")
            f.write(f"IMU dt std: {np.std(dt_imu)*1000:.3f} ms\n")
            f.write(f"IMU dt min/max: {np.min(dt_imu)*1000:.3f} / {np.max(dt_imu)*1000:.3f} ms\n\n")
            f.write(f"Acc mean:  [{np.mean(imu[:,1]):.6f}, {np.mean(imu[:,2]):.6f}, {np.mean(imu[:,3]):.6f}]\n")
            f.write(f"Acc std:   [{np.std(imu[:,1]):.6f}, {np.std(imu[:,2]):.6f}, {np.std(imu[:,3]):.6f}]\n")
            f.write(f"Gyr mean:  [{np.mean(imu[:,4]):.6f}, {np.mean(imu[:,5]):.6f}, {np.mean(imu[:,6]):.6f}]\n")
            f.write(f"Gyr std:   [{np.std(imu[:,4]):.6f}, {np.std(imu[:,5]):.6f}, {np.std(imu[:,6]):.6f}]\n\n")
        else:
            f.write("IMU: NO DATA RECEIVED\n\n")

        if tdoa_data:
            tdoa = np.array(tdoa_data)
            dt_tdoa = np.diff(tdoa[:, 0])
            f.write(f"TDOA samples: {len(tdoa_data)}\n")
            f.write(f"TDOA duration: {tdoa[-1, 0] - tdoa[0, 0]:.3f} s\n")
            f.write(f"TDOA rate: {1.0 / np.mean(dt_tdoa):.1f} Hz (mean dt={np.mean(dt_tdoa)*1000:.3f} ms)\n")
            f.write(f"TDOA dt min/max: {np.min(dt_tdoa)*1000:.3f} / {np.max(dt_tdoa)*1000:.3f} ms\n\n")

            # Per-pair statistics
            f.write("Per anchor-pair TDOA stats:\n")
            f.write(f"{'idA':>4} {'idB':>4} {'count':>6} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}\n")
            pairs = {}
            for row in tdoa_data:
                key = (int(row[1]), int(row[2]))
                pairs.setdefault(key, []).append(row[3])
            for (a, b) in sorted(pairs.keys()):
                vals = np.array(pairs[(a, b)])
                f.write(f"{a:4d} {b:4d} {len(vals):6d} {np.mean(vals):10.4f} "
                        f"{np.std(vals):10.4f} {np.min(vals):10.4f} {np.max(vals):10.4f}\n")
            f.write("\n")

            # Cycle structure analysis
            f.write("Cycle structure (first 200 TDOA messages):\n")
            for i, row in enumerate(tdoa_data[:200]):
                f.write(f"  [{i:4d}] t={row[0]:.6f} pair=({int(row[1]):d},{int(row[2]):d}) tdoa={row[3]:.4f}\n")
        else:
            f.write("TDOA: NO DATA RECEIVED\n\n")

    rospy.loginfo("Saved to /tmp/imu_raw.csv, /tmp/tdoa_raw.csv, /tmp/data_summary.txt")
    sys.exit(0)

if __name__ == "__main__":
    rospy.init_node("dump_raw_data", anonymous=True)

    # Use same topic names as your launch file
    rospy.Subscriber("/imu_data", Imu, imu_cb, queue_size=2000)
    rospy.Subscriber("/tdoa_data", Tdoa, tdoa_cb, queue_size=2000)

    signal.signal(signal.SIGINT, save_and_exit)
    rospy.on_shutdown(lambda: save_and_exit())

    rospy.loginfo("Recording IMU + TDOA data. Press Ctrl+C to stop and save...")
    rospy.spin()