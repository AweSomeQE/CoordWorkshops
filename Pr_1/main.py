import numpy as np
import time

# Functions for Cartesian and Polar coordinate conversions and distances
def cartesian_to_polar(cartesian_points):
    x, y = cartesian_points[:, 0], cartesian_points[:, 1]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return np.stack((r, theta), axis=1)

def polar_to_cartesian(polar_points):
    r, theta = polar_points[:, 0], polar_points[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack((x, y), axis=1)

def cartesian_to_spherical(cartesian_points):
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.linalg.norm(cartesian_points, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack((r, theta, phi), axis=1)

def spherical_to_cartesian(spherical_points):
    r, theta, phi = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack((x, y, z), axis=1)

def distance_cartesian(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def distance_polar(p1, p2):
    delta_theta = np.cos(p1[:, 1] - p2[:, 1])
    dist = np.sqrt(p1[:, 0]**2 + p2[:, 0]**2 - 2 * p1[:, 0] * p2[:, 0] * np.clip(delta_theta, -1, 1))
    return dist

def distance_spherical(p1, p2):
    phi_diff = np.cos(p1[:, 2] - p2[:, 2])
    angle_term = np.sin(p1[:, 1]) * np.sin(p2[:, 1]) * phi_diff + np.cos(p1[:, 1]) * np.cos(p2[:, 1])
    dist = np.sqrt(p1[:, 0]**2 + p2[:, 0]**2 - 2 * p1[:, 0] * p2[:, 0] * np.clip(angle_term, -1, 1))
    return dist

# Array shuffling helper function
def shuffle_array(arr):
    np.random.shuffle(arr)
    return arr

# Benchmarking various operations with detailed output
def benchmark(num_points):
    print(f"Benchmarking with {num_points} random points\n")

    # Benchmarking 2D Cartesian to Polar conversion
    points_2d = np.random.rand(num_points, 2)
    start = time.perf_counter()
    polar_points = cartesian_to_polar(points_2d)
    elapsed = time.perf_counter() - start
    print(f"  - Time for 2D Cartesian to Polar conversion: {elapsed:.6f} seconds")

    # Benchmarking 2D Polar to Cartesian conversion
    start = time.perf_counter()
    cartesian_points = polar_to_cartesian(polar_points)
    elapsed = time.perf_counter() - start
    print(f"  - Time for 2D Polar to Cartesian conversion: {elapsed:.6f} seconds")

    # Benchmarking 3D Cartesian to Spherical conversion
    points_3d = np.random.rand(num_points, 3)
    start = time.perf_counter()
    spherical_points = cartesian_to_spherical(points_3d)
    elapsed = time.perf_counter() - start
    print(f"  - Time for 3D Cartesian to Spherical conversion: {elapsed:.6f} seconds")

    # Benchmarking 3D Spherical to Cartesian conversion
    start = time.perf_counter()
    cartesian_points_3d = spherical_to_cartesian(spherical_points)
    elapsed = time.perf_counter() - start
    print(f"  - Time for 3D Spherical to Cartesian conversion: {elapsed:.6f} seconds")

    # Benchmarking distance calculations for 2D Cartesian points
    start = time.perf_counter()
    cartesian_distances = distance_cartesian(points_2d, shuffle_array(points_2d.copy()))
    elapsed = time.perf_counter() - start
    print(f"\n  - Time for 2D Cartesian distance calculation: {elapsed:.6f} seconds")

    # Benchmarking distance calculations for Polar points
    start = time.perf_counter()
    polar_distances = distance_polar(polar_points, shuffle_array(polar_points.copy()))
    elapsed = time.perf_counter() - start
    print(f"  - Time for Polar distance calculation: {elapsed:.6f} seconds")

    # Benchmarking distance calculations for Spherical points
    start = time.perf_counter()
    spherical_distances = distance_spherical(spherical_points, shuffle_array(spherical_points.copy()))
    elapsed = time.perf_counter() - start
    print(f"  - Time for Spherical distance calculation: {elapsed:.6f} seconds")

# Function to run benchmark and display header
def print_benchmark(num_points):
    print(f"Running benchmark with {num_points} points")
    benchmark(num_points)

# Run benchmark for varying numbers of points
for i in range(1, 7):
    print_benchmark(10 ** i)
