import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import LineString
from math import atan2, cos, sin, sqrt
from scipy.optimize import minimize

# Distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Trilateration to estimate position based on distances
def trilaterate(base_stations, distances):
    p1, p2, p3 = base_stations
    d1, d2, d3 = distances

    ex = (np.array(p2) - np.array(p1)) / np.linalg.norm(np.array(p2) - np.array(p1))
    i = np.dot(ex, np.array(p3) - np.array(p1))
    ey = (np.array(p3) - np.array(p1) - i * ex) / np.linalg.norm(np.array(p3) - np.array(p1) - i * ex)
    ez = np.cross(ex, ey)

    d = np.linalg.norm(np.array(p2) - np.array(p1))
    j = np.dot(ey, np.array(p3) - np.array(p1))

    x = (d1**2 - d2**2 + d**2) / (2 * d)
    y = ((d1**2 - d3**2 + i**2 + j**2) / (2 * j)) - (i / j) * x
    z = sqrt(abs(d1**2 - x**2 - y**2))

    position = np.array(p1) + x * ex + y * ey
    return position[:2]


# Gradient descent function 
def objective_function(guess, base_stations, measured_distances):
    estimated_distances = [calculate_distance(guess, bs) for bs in base_stations]
    error = np.sum((np.array(estimated_distances) - np.array(measured_distances))**2)
    return error


# Define base stations and object coordinates
base_stations = [
    [0, 0],  # Base station 1
    [10, 0],  # Base station 2
    [5, 10]  # Base station 3
]

true_object_position = [3, 4]

# Distances from base stations to the object
true_distances = [calculate_distance(true_object_position, bs) for bs in base_stations]

# Estimate object's position without noise
trilaterated_position = trilaterate(base_stations, true_distances)

# Gradient descent to find object position without noise
initial_guess = [0, 0]
result = minimize(
    objective_function,
    initial_guess,
    args=(base_stations, true_distances),
    method="Powell",
)

# Define lines for triangulation
true_angles = [
    atan2(true_object_position[1] - base_stations[0][1], true_object_position[0] - base_stations[0][0]),
    atan2(true_object_position[1] - base_stations[1][1], true_object_position[0] - base_stations[1][0]),
    atan2(true_object_position[1] - base_stations[2][1], true_object_position[0] - base_stations[2][0])
]

lines = [
    LineString([base_stations[0], [base_stations[0][0] + 10 * cos(true_angles[0]), base_stations[0][1] + 10 * sin(true_angles[0])]]),
    LineString([base_stations[1], [base_stations[1][0] + 10 * cos(true_angles[1]), base_stations[1][1] + 10 * sin(true_angles[1])]]),
    LineString([base_stations[2], [base_stations[2][0] + 10 * cos(true_angles[2]), base_stations[2][1] + 10 * sin(true_angles[2])]])
]

# Find intersection point for triangulation
intersection = lines[0].intersection(lines[1])
triangulated_position = (intersection.xy[0][0], intersection.xy[1][0]) if not intersection.is_empty else None


# Add noise to the distances and recompute
noise_level = 0.5  # Adjust noise level here
noisy_distances = [d + np.random.normal(0, noise_level) for d in true_distances]

# Trilaterate with noise
trilaterated_position_noisy = trilaterate(base_stations, noisy_distances)

# Gradient descent with noise
result_noisy = minimize(
    objective_function,
    initial_guess,
    args=(base_stations, noisy_distances),
    method="Powell",
)

# Find triangulated position with noise
triangulated_with_noise = None
if not intersection.is_empty:
    angle_offsets = [np.random.normal(0, noise_level) for _ in range(3)]
    noisy_angles = [true_angles[i] + angle_offsets[i] for i in range(3)]

    noisy_lines = [
        LineString([base_stations[0], [base_stations[0][0] + 10 * cos(noisy_angles[0]), base_stations[0][1] + 10 * sin(noisy_angles[0])]]),
        LineString([base_stations[1], [base_stations[1][0] + 10 * cos(noisy_angles[1]), base_stations[1][1] + 10 * sin(noisy_angles[1])]]),
        LineString([base_stations[2], [base_stations[2][0] + 10 * cos(noisy_angles[2]), base_stations[2][1] + 10 * sin(noisy_angles[2])]])
    ]

    noisy_intersection = noisy_lines[0].intersection(noisy_lines[1])

    if not noisy_intersection.is_empty:
        triangulated_with_noise = (noisy_intersection.xy[0][0], noisy_intersection.xy[1][0])

# Visualization
plt.figure(figsize=(12, 12))  
plt.plot([bs[0] for bs in base_stations], [bs[1] for bs in base_stations], 'ko', label='Base Stations')  # Base stations as black dots
plt.plot(true_object_position[0], true_object_position[1], 'r*', label='True Object Position')  # Origin as a star

# Visualize trilaterated position and gradient descent result without noise
plt.plot(trilaterated_position[0], trilaterated_position[1], 'b^', label='Trilateration without noise')  
plt.plot(result.x[0], result.x[1], 'bs', label='Gradient Descent without noise')  

# Visualize trilaterated position and gradient descent result with noise
plt.plot(trilaterated_position_noisy[0], trilaterated_position_noisy[1], 'g^', label='Trilateration with noise')  
plt.plot(result_noisy.x[0], result_noisy.x[1], 'gs', label='Gradient Descent with noise')  

# Visualize triangulated position and gradient descent result with noise
if triangulated_position:
    plt.plot(triangulated_position[0], triangulated_position[1], 'y+', label='Triangulation without noise')  # Yellow cross for triangulation without noise
if triangulated_with_noise:
    plt.plot(triangulated_with_noise[0], triangulated_with_noise[1], 'y*', label='Triangulation with noise')  # Yellow star for triangulation with noise

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Base Stations and Object Positions")
plt.legend()
plt.show()

# Console output without noise
print("Results without noise:")
print(f"  Trilaterated Position: {trilaterated_position}")
print(f"  Gradient Descent Position: {result.x}")
print(f"  Triangulated Position: {triangulated_position}")

# Console output with noise
print("\nResults with noise:")
print(f"  Noise Level: {noise_level}")
print(f"  Trilaterated Position: {trilaterated_position_noisy}")
print(f"  Gradient Descent Position: {result_noisy.x}")
print(f"  Triangulated Position with noise: {triangulated_with_noise if triangulated_with_noise else 'No valid intersection found.'}")
