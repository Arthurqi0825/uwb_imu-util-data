import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def analyze_trajectory(file_path):
    try:
        # 1. Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)
    
    # 2. Define statistical helper
    def get_stats(error_series):
        return {
            'Mean (m)': error_series.mean(),
            'Median (m)': error_series.median(),
            'RMSE (m)': np.sqrt((error_series**2).mean()),
            'Max (m)': error_series.max(),
            '95th% (m)': np.percentile(error_series, 95)
        }

    # 3. Calculate Statistics
    wls_stats = get_stats(df['wls_gt_err'])
    tc_stats = get_stats(df['tc_gt_err'])
    
    stats_df = pd.DataFrame([wls_stats, tc_stats], 
                            index=['WLS (UWB Only)', 'TC (UWB+IMU)'])
    
    print("\n--- Performance Metrics ---")
    print(stats_df.to_string())

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Trajectory (XY Plane)
    axes[0].plot(df['gt_x'], df['gt_y'], 'k-', label='Ground Truth', linewidth=2)
    axes[0].plot(df['wls_x'], df['wls_y'], 'r.', markersize=2, alpha=0.3, label='WLS')
    axes[0].plot(df['tc_x'], df['tc_y'], 'b--', label='TC (Fused)', alpha=0.8)
    axes[0].set_title('XY Trajectory')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')

    # Plot B: Error over Time
    axes[1].plot(df['timestamp'], df['wls_gt_err'], 'r', alpha=0.4, label='WLS Error')
    axes[1].plot(df['timestamp'], df['tc_gt_err'], 'b', alpha=0.7, label='TC Error')
    axes[1].set_title('Positioning Error over Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error (m)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot C: CDF of Errors
    sorted_wls = np.sort(df['wls_gt_err'])
    y_wls = np.arange(len(sorted_wls)) / float(len(sorted_wls)-1)
    sorted_tc = np.sort(df['tc_gt_err'])
    y_tc = np.arange(len(sorted_tc)) / float(len(sorted_tc)-1)

    axes[2].plot(sorted_wls, y_wls, 'r', label='WLS')
    axes[2].plot(sorted_tc, y_tc, 'b', label='TC')
    axes[2].set_title('Error CDF')
    axes[2].set_xlabel('Error (m)')
    axes[2].set_ylabel('Probability')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='Analyze UWB and IMU trajectory fusion results from a CSV file.')
    
    # Add positional argument for the file path
    parser.add_argument('filename', type=str, help='Path to the input CSV file (e.g., trajectory_data.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis
    analyze_trajectory(args.filename)