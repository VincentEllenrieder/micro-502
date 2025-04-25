import numpy as np
import time
import cv2
from lib.simple_pid import PID
import matplotlib.pyplot as plt # to erase for assignment
from mpl_toolkits.mplot3d import Axes3D # to erase for assignment
from itertools import permutations, product


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

# NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
# If you want to display the camera image you can call it main.py.

MANEUVER = {
    # --- Take off ---
    "Started" : 0,
    # --- Gate search ---
    "Search next gate": 0,
    "Go to scan" :0,
    "Vision" : 0,
    # --- Go to gate ---
    "Go to gate" : 0,
    # --- Race ---
    "Start race" : 0,
    "Race" : 0
}

# Gates data
N_GATES = 5                                                              # Number of gates in the map 
GATES_DATA = {
    f"GATE{i+1}": {
        "centroid": None,                                               # Triangulated centroid position, ndarray of shape (3,)
        "corners": [],                                                  # Triangulated corner positions, list of 4 ndarrays of shape (3,)
        "normal points": []                                             # Normal points on each side of the gate, list of 2 ndarrays of shape (3,)
    }
    for i in range(N_GATES)
}
gates_found = 0
N_CORNERS = 4                                                          # Number of corners to detect in the image
DISTANCE_FROM_GATE = 0.5                                               # Distance of the normal point from the centroid when going through a gate

# Camera parameters
FOCAL_LENGTH = 161.013922282    # Focal length of the camera in pixels
X_CAM = 0.030                   # x position of the camera relative to the body frame in meters
Y_CAM = 0.000                   # y position of the camera relative to the body frame in meters
Z_CAM = 0.010                   # z position of the camera relative to the body frame in meters

# Thresholds for pink gate detection
RED_BLUE_THRESHOLD_HIGH = 230   # 190 < R, B < 230
RED_BLUE_THRESHOLD_LOW = 190 
GREEN_THRESHOLD = 180           # G < 180 

# Start position of the drone
X_START = 1.0
Y_START = 4.0
Z_START = 1.3

# Scan locations
R = 0.5 # Radius of the scan circle
CENTER_X = 4.0 # X position of the map center
CENTER_Y = 4.0 # Y position of the map center
SCAN_LOCATIONS = {
    "SCAN1" : {
        "SCAN_X" : R*np.cos(-2*np.pi/3) + CENTER_X, 
        "SCAN_Y" : R*np.sin(-2*np.pi/3) + CENTER_Y,
        "SCAN_YAW" : -2*np.pi/3
    },
    "SCAN2" : {
        "SCAN_X" : R*np.cos(-np.pi/3) + CENTER_X,
        "SCAN_Y" : R*np.sin(-np.pi/3) + CENTER_Y,
        "SCAN_YAW" : -np.pi/3
    },
    "SCAN3" : {
        "SCAN_X" : R*np.cos(0) + CENTER_X,
        "SCAN_Y" : R*np.sin(0) + CENTER_Y,
        "SCAN_YAW" : 0
    },
    "SCAN4" : {
        "SCAN_X" : R*np.cos(np.pi/3) + CENTER_X,
        "SCAN_Y" : R*np.sin(np.pi/3) + CENTER_Y,
        "SCAN_YAW" : np.pi/3
    },
    "SCAN5" : {
        "SCAN_X" : R*np.cos(2*np.pi/3) + CENTER_X,
        "SCAN_Y" : R*np.sin(2*np.pi/3) + CENTER_Y,
        "SCAN_YAW" : 2*np.pi/3
    }
}
POS_MARGIN = 0.05
YAW_MARGIN = np.pi/16

# Drone placement paramenters for picture taking in 1st lap
N_IMAGES = 2                                                            # Number of images to take for triangulation
images_taken = 0                                                        # Number of images taken for triangulation
ALIGNMENT_MARGIN = 15                                                   # Margin to consider drone aligned with gate centroid
CENTROID_YAW_PID = PID(Kp=0.005, Ki=0.0003, Kd=0.001, output_limits=(-np.pi/20, np.pi/20))   # PID controller for centroid alignment
CENTROID_YAW_PID.setpoint = 0                                           # Alignment in Y should be at center of image  
no_features = False                                                     # Flag to check if no centroid was found in the image  
aligned = False                                                         # Flag to check if drone has aligned with gate centroid
deviated = False                                                        # Flag to check if drone has deviated for second image
current_image_corners = None                                            # Current image corners of the gate


# Triangulation parameters
R_C2B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])                   # Rotation matrix from camera frame to body frame 
r_s_vectors = np.zeros((3, N_CORNERS+1, N_IMAGES))                      # Image vectors in inertial frame of the 4 corners + centroid of the current gate taken in images 1 and 2
p_q_vectors = np.zeros((3, N_CORNERS+1, N_IMAGES))                      # Camera positions in inertial frame when 4 corners + centroid of the current gate were taken in images 1 and 2 

# Move to gate parameters
reached_normalpt1 = False                                               # Flag to check if the first normal point of the gate has been reached
reached_normalpt2 = False                                               # Flag to check if the second normal point of the gate has been reached

# Racing parameters
VEL_LIM = 7.0                                                           # Velocity limit in m/s
ACC_LIM = 50.0                                                          # Acceleration limit in m/s^2
DISC_STEPS = 20                                                         # Number of discrete steps per segment for the trajectory planning
T_FINAL = 20                                                            # Time to finish both racing laps in seconds
ANGLE_PENALTY = 1.0                                                     # Penalty for path choice with high turning angles

# General purpose registers
displacement_goal = np.zeros(3)   # Saved current displacement goal

# ----------------- Main function -----------------

def get_command(sensor_data, camera_data, dt):
    
    global MANEUVER, images_taken, gates_found, no_features, aligned, deviated, displacement_goal, current_image_corners, \
                    r_s_vectors, p_q_vectors, GATES_DATA, reached_normalpt1, reached_normalpt2, \
                    race_waypoints, race_indices, race_poly_coeffs, race_trajectory_setpoints, race_time_setpoints, race_times, race_time

    # Drone has not taken off
    if MANEUVER["Started"] == 0 :
        print('in Started', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])

        if sensor_data['z_global'] < Z_START :
            control_command = [sensor_data['x_global'], sensor_data['y_global'], Z_START, sensor_data['yaw']]
            return control_command 

        else:
            MANEUVER['Started'] = 1
            MANEUVER["Search next gate"] = 1
            MANEUVER["Go to scan"] = 1
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command 
    
    # Search next gate
    elif MANEUVER["Search next gate"] == 1 :
        print('in Search next gate', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])
        
        if MANEUVER["Go to scan"] == 1:
            print('in Go to scan')
            scan_location = SCAN_LOCATIONS["SCAN" + str(gates_found + 1)] # Get the next scan location

            # If at scan location, go to next step
            if (scan_location["SCAN_X"] - POS_MARGIN < sensor_data['x_global'] < scan_location["SCAN_X"] + POS_MARGIN) and \
               (scan_location["SCAN_Y"] - POS_MARGIN < sensor_data['y_global'] < scan_location["SCAN_Y"] + POS_MARGIN) and \
               (scan_location["SCAN_YAW"] - YAW_MARGIN < sensor_data['yaw'] < scan_location["SCAN_YAW"] + YAW_MARGIN) and \
               (Z_START - POS_MARGIN < sensor_data['z_global'] < Z_START + POS_MARGIN) :
                print("At scan location")
                MANEUVER["Go to scan"] = 0
                MANEUVER["Vision"] = 1
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command 

            # Go to scan location if not there yet
            else :
                print("Not at scan location")
                control_command = [scan_location["SCAN_X"], scan_location["SCAN_Y"], Z_START, scan_location["SCAN_YAW"]]
                return control_command
            
        elif MANEUVER["Vision"] == 1:
            print('in Vision')
            
            while images_taken < N_IMAGES:
                
                if not no_features:
                    centroid, corners = get_centroid_and_corners(camera_data)
                    print("Centroid: ", centroid)
                    if (centroid is None) or (corners is None): # if no controid or corners found, prepare to displace
                        print("No centroid or corners found")
                        no_features = True
                        deviation = 0.0
                        if gates_found == 2:
                            deviation = 0.2 # If searching for third gate, y displacement will be 0 if no deviation set -> set some to have good sight of gate
                        displacement_goal = displacement([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']], sensor_data['yaw'], move_radius=0.25, deviation=deviation, in_yaw_direction=0) # Displace in direction of yaw when aligned to centroid
                        print("Setting displacement goal after no centroid or corners were found: ", displacement_goal)

                # If no features found (orthogonal alignment with gate or very narrow contour) do :
                if no_features :
                    # If at displacement goal, stop displacement maneuver
                    if (displacement_goal[0] - POS_MARGIN < sensor_data['x_global'] < displacement_goal[0] + POS_MARGIN) and \
                       (displacement_goal[1] - POS_MARGIN < sensor_data['y_global'] < displacement_goal[1] + POS_MARGIN) :
                        print("At displacement goal after no centroid or corners were found")
                        no_features = False
                        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                        return control_command
                    # If not at displacement goal, displace
                    else :
                        print("Moving to displacement goal after no centroid or corners were found")
                        control_command = displacement_goal
                        return control_command
                
                # If 1 image already taken (gate found), align drone to the centroid of the gate and displace before taking the next image for better triangulation
                if images_taken != 0:
                    cX, _ = centroid
                    if (abs(cX) > ALIGNMENT_MARGIN) and (not aligned): # Align to centroid of the gate
                        print("Aligning, error = ", cX)
                        yaw_correction = CENTROID_YAW_PID(cX)
                        print("Yaw correction: ", yaw_correction)
                        new_yaw = sensor_data['yaw'] + yaw_correction
                        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], new_yaw]
                        return control_command
                    CENTROID_YAW_PID.reset()    # Anti-windup reset                     
                    aligned = True

                    if not ((displacement_goal[0] - POS_MARGIN < sensor_data['x_global'] < displacement_goal[0] + POS_MARGIN) and \
                            (displacement_goal[1] - POS_MARGIN < sensor_data['y_global'] < displacement_goal[1] + POS_MARGIN)) and (not deviated) :
                        print("Deviating")
                        control_command = displacement_goal
                        return control_command
                    deviated = True

                # Save corners and centroid image vectors and camera positions in inertial frame
                current_image_corners = corners.copy()
                for t in range(N_CORNERS): # Triangulate the 4 corners
                    corner = tuple(corners[t, :])
                    r, p = triangulation_preprocess(corner, sensor_data)
                    r_s_vectors[:, t, images_taken] = r
                    p_q_vectors[:, t, images_taken] = p

                r, p = triangulation_preprocess(centroid, sensor_data)
                r_s_vectors[:, -1, images_taken] = r
                p_q_vectors[:, -1, images_taken] = p

                # Compute slope of the top corners to displace in the right direction
                slope = compute_corner_slope(current_image_corners) 
                if slope == 1:
                    body_x = 0.28
                    body_y = 0.28
                    body_z = 0
                elif slope == -1:
                    body_x = 0.28
                    body_y = -0.28
                    body_z = 0
                R_b2i = quaternion2rotmat([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
                displacement_goal = R_b2i @ np.array([body_x, body_y, body_z]) + np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
                displacement_goal = np.append(displacement_goal, sensor_data['yaw'])
                print("Second position: ", displacement_goal)

                images_taken += 1
                print("Image ", images_taken, " taken")

            # Triangulate the 4 corners
            for t in range(N_CORNERS): 
                corner_position = triangulation(t)
                corner_position = np.clip(corner_position, 0.1, 8.0) # Clip the position to avoid inconsistent values
                print("Corner", t, " position: ", corner_position)
                GATES_DATA[f"GATE{gates_found+1}"]["corners"].append(corner_position)
            
            # Triangulate the centroid
            gate_position = triangulation(-1)
            gate_position = np.clip(gate_position, 0.1, 8.0) # Clip the position to avoid inconsistent values
            print("Gate position: ", gate_position)
            GATES_DATA[f"GATE{gates_found+1}"]["centroid"] = gate_position

            # Reset flags and registers for next gate
            deviated = False
            aligned = False
            images_taken = 0
            r_s_vectors = np.zeros((3, N_CORNERS+1, N_IMAGES))                                   
            p_q_vectors = np.zeros((3, N_CORNERS+1, N_IMAGES))   

            # Compute the normal points to go through the gate
            normal = plane_normal(GATES_DATA[f"GATE{gates_found+1}"]["corners"])
            centroid = GATES_DATA[f"GATE{gates_found+1}"]["centroid"]
            np1, np2 = normal_points(normal, centroid)
            np1 = np.clip(np1, 0.1, 8.0) # Clip the positions to avoid inconsistent values
            np2 = np.clip(np2, 0.1, 8.0) 
            GATES_DATA[f"GATE{gates_found+1}"]["normal points"].append(np1)
            GATES_DATA[f"GATE{gates_found+1}"]["normal points"].append(np2)
            print("Normal points: ", GATES_DATA[f"GATE{gates_found+1}"]["normal points"])

            # Prepare to go to the gate found
            MANEUVER["Vision"] = 0
            MANEUVER["Search next gate"] = 0
            MANEUVER["Go to gate"] = 1
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command 
    
    elif MANEUVER["Go to gate"] == 1:
        print('in Go to gate', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])
        
        # If drone hasn't reached the first normal point yet, go to it
        if not reached_normalpt1:
            x_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][0][0]
            y_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][0][1]
            z_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][0][2]
            goal = displacement([x_goal, y_goal, z_goal], sensor_data['yaw'], move_radius=0, deviation=0) 
            print("Point 1 goal: ", goal)
            
        # If drone has reached the first normal point but not the second, go to the second normal point
        elif (reached_normalpt1) and (not reached_normalpt2):
            x_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][1][0]
            y_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][1][1]
            z_goal = GATES_DATA[f"GATE{gates_found+1}"]["normal points"][1][2]
            goal = displacement([x_goal, y_goal, z_goal], sensor_data['yaw'], move_radius=0, deviation=0)
            print("Point 2 goal: ", goal)

        # If the drone went to both points, it crossed the gate -> prepare for next step
        else :
            reached_normalpt1 = False
            reached_normalpt2 = False
            gates_found += 1
            if gates_found < N_GATES:
                MANEUVER["Go to gate"] = 0
                MANEUVER["Search next gate"] = 1
                MANEUVER["Go to scan"] = 1
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command
            else :
                # visualize_gates(GATES_DATA) # to erase for assignment
                MANEUVER["Go to gate"] = 0
                MANEUVER["Start race"] = 1
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command

        if (goal[0] - POS_MARGIN < sensor_data['x_global'] < goal[0] + POS_MARGIN) and \
           (goal[1] - POS_MARGIN < sensor_data['y_global'] < goal[1] + POS_MARGIN) and \
           (goal[2] - POS_MARGIN < sensor_data['z_global'] < goal[2] + POS_MARGIN) and \
           (goal[3] - YAW_MARGIN < sensor_data['yaw'] < goal[3] + YAW_MARGIN) :
            if not reached_normalpt1:
                print("At normal point 1")
                reached_normalpt1 = True
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command
            elif (reached_normalpt1) and (not reached_normalpt2):
                print("At normal point 2")
                reached_normalpt2 = True
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command
        else:
            print("Moving to gate position")
            control_command = goal
            return control_command

    elif MANEUVER['Start race'] == 1:
        print('in Start race', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])
        if (X_START - POS_MARGIN < sensor_data['x_global'] < X_START + POS_MARGIN) and \
           (Y_START - POS_MARGIN < sensor_data['y_global'] < Y_START + POS_MARGIN) and \
           (Z_START - POS_MARGIN < sensor_data['z_global'] < Z_START + POS_MARGIN) :
            print("At start position")
            
            # Initialize the order of the waypoints used for racing and each segment time
            wp = []
            for i in range(N_GATES):  # Assuming 5 gates
                np1, np2 = GATES_DATA[f"GATE{i+1}"]["normal points"]
                wp.extend([np1, np2])  # flat list of 10 points (np.array)

            best_wp_order, best_wp_indices, min_cost = sort_wp_min_energy(wp)
            print("Best waypoint order: ", best_wp_indices, "\nCost: ", min_cost)

            best_wp_order_with_centroids = []
            # Add in the waypoints order the centrois of the gate between each pair of normal points of that gate
            for i in range(0, len(best_wp_indices), 2):
                idx1 = best_wp_indices[i]

                # Append first normal point
                best_wp_order_with_centroids.append(best_wp_order[i])

                # Identify which gate this pair belongs to
                gate_num = idx1 // 2  # Because two normal points per gate (0,1 → gate 0; 2,3 → gate 1)

                # Append the centroid of that gate
                centroid = GATES_DATA[f"GATE{gate_num+1}"]["centroid"]
                best_wp_order_with_centroids.append(centroid)

                # Append second normal point
                best_wp_order_with_centroids.append(best_wp_order[i+1])
                
            best_path = best_wp_order_with_centroids
            best_path.insert(0, [X_START, Y_START, Z_START])  # Add start position
            best_path.extend(best_wp_order_with_centroids)    # Add points for a second lap
            best_path.extend([[X_START, Y_START, Z_START]])   # Add final position

            race_waypoints = best_path
            race_indices = best_wp_indices

            segment_times = init_params(race_waypoints)
            race_times = np.concatenate(([0], np.cumsum(segment_times)))

            # Compute polynomial coefficients and extract trajectory setpoints (sampled points)
            race_poly_coeffs = compute_poly_coefficients(race_waypoints, race_times)
            race_trajectory_setpoints, race_time_setpoints  = poly_setpoint_extraction(race_poly_coeffs, race_times)

            # Prepare for racing
            race_time = 0.0
            MANEUVER['Start race'] = 0
            MANEUVER['Race'] = 1
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
        else:
            print("Moving to starting block to begin racing")
            control_command = [X_START, Y_START, Z_START, sensor_data['yaw']]
            return control_command

    elif MANEUVER['Race'] == 1:
        print('in Race', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])

        # Update race timer
        race_time += dt

        final_target = race_trajectory_setpoints[-1][:3]  # Extract [x, y, z] of last setpoint
        current_pos = np.array([
            sensor_data['x_global'],
            sensor_data['y_global'],
            sensor_data['z_global']
        ])
        # End race if finished
        if (np.linalg.norm(final_target - current_pos) < POS_MARGIN) and (race_time + 5 > T_FINAL):
            print("Race finished! Time taken:", race_time)
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
        
        # Choose current setpoint
        else:
            idx = np.searchsorted(race_time_setpoints, race_time)
            idx = min(idx, len(race_trajectory_setpoints)-1)
            target = race_trajectory_setpoints[idx]  # [x, y, z, yaw]

            control_command = [float(target[0]), float(target[1]), float(target[2]), float(target[3])]
            return control_command
      
# ----------------- Helper functions for first vision based lap -----------------

def image_processing(camera_data):
    img = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2RGB) # convert BGRA to RGB
    plt.imsave('image_analysis/original_image.png', img[:, :, :])
    # Mask the image to only keep the pink area
    img_filtered = np.zeros(img.shape[:2], dtype=np.uint8)
    condition = (img[:, :, 0] > RED_BLUE_THRESHOLD_LOW) & (img[:, :, 0] < RED_BLUE_THRESHOLD_HIGH) & \
                (img[:, :, 2] > RED_BLUE_THRESHOLD_LOW) & (img[:, :, 2] < RED_BLUE_THRESHOLD_HIGH) & \
                (img[:, :, 1] < GREEN_THRESHOLD)
    img_filtered[condition] = 255
    plt.imsave('image_analysis/thresholding_image.png', img_filtered[:, :], cmap='gray')
    return img_filtered

def get_centroid_and_corners(camera_data):
    image_filtered = image_processing(camera_data)
    image_features = np.zeros((*np.shape(image_filtered), 3), dtype=np.uint8) 
    contours, _ = cv2.findContours(image_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Find contours in the image
    #print('contours \n :', contours, ', len :', len(contours), ', shape :', np.shape(contours), ', type :', type(contours)) 
    
    # If sight orthogonal to gate's normal vector (no countours), signal to displace 
    if len(contours) == 0:
        print("No contours found")
        return None, None
    
    # If one contour found, directly compute its centroid
    elif len(contours) == 1:
        contour = contours[0]

        cv2.drawContours(image_features, [contour], 0, (0, 255, 0), 1)
        plt.imsave('image_analysis/gate_single_contour' + str(images_taken+1) + '.png', image_features[:, :, :])          

        result_centroid = compute_centroid(contour)      
        if result_centroid is None:
            return None, None
        cX = result_centroid[0]
        cY = result_centroid[1]
        centroid = cX - 150, cY - 150                   # Shift centroid coordinates wrt to center of image

        result_corners = compute_corners(contour)
        if result_corners is None:
            return None, None
        corners = sort_corners(result_corners) - 150    # Sort corners in order and shift coordinates wrt to center of image
        
        image_features[cY, cX, 0] = 255                 # Set the centroid pixel to red
        for p, (x, y) in enumerate(corners):            # Set corners in shades of blue
            xi, yi = int(x), int(y)
            #color_coeff = p % len(corners)     
            image_features[yi + 150, xi + 150, 2] = 255
        plt.imsave('image_analysis/gate_features_single_contour_image' + str(images_taken+1) + '.png', image_features[:, :, :])

        return centroid, corners                                                           

    # If multiple contours found, only take the one with rightmost centroid
    elif len(contours) > 1:

        cv2.drawContours(image_features, contours, -1, (0, 255, 0), 1)
        plt.imsave('image_analysis/gate_multiple_contour' + str(images_taken+1) + '.png', image_features[:, :, :])          

        rightmost = 0
        contour_chosen = None
        cX, cY = 0, 0
        for contour in contours:
            result_centroid = compute_centroid(contour)
            if result_centroid is None:
                return None, None
            x, y = result_centroid
            if x > rightmost:
                rightmost = x                           # Take the rightmost centroid
                cX, cY = x - 150, y - 150               # Shift centroid to center of image
                centroid = cX, cY
                contour_chosen = contour
                
        result_corners = compute_corners(contour_chosen)
        if result_corners is None:
            return None, None
        corners = sort_corners(result_corners) - 150    # Sort corners in order and shift coordinates wrt to center of image

        image_features[cY, cX, 0] = 255                 # Set the centroid pixel to red
        for p, (x, y) in enumerate(corners):            # Set corners in shades of blue
            xi, yi = int(x), int(y)
            #color_coeff = p % len(corners)     
            image_features[yi + 150, xi + 150, 2] = 255
        plt.imsave('image_analysis/gate_features_multiple_contours' + str(images_taken+1) + '.png', image_features[:, :, :])

        return centroid, corners

def compute_centroid(contour):
    # Compute the centroid of the given contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        print("Error computing centroid")
        return None

def compute_corners(contour):
    # Approximate contour to polygon
    perim = cv2.arcLength(contour, True)
    for factor in np.linspace(0.001, 0.1, 100):
        eps = factor * perim
        corners = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(corners) == 4:
            return corners
    print("Contour does not have 4 corners")
    return None

def sort_corners(corners):
    """
    Sort the corners in a specific order: top-left, top-right, bottom-right, bottom-left

    Inputs: 
                (4, 2) np array of corner points
    Outputs: 
                (4, 2) np array of sorted corner points
    """

    x_sorted = corners[np.argsort(corners[:, 0]), :]    # Sort by increasing x

    left = x_sorted[:2, :]                              # Two leftmost points
    right = x_sorted[2:, :]                             # Two rightmost points

    left = left[np.argsort(left[:, 1]), :]              # Among left points, sort by y to get top-left and bottom-left
    top_left, bottom_left = left                    

    right = right[np.argsort(right[:, 1]), :]           # Among right points, sort by y to get top-left and bottom-left
    top_right, bottom_right = right

    return np.array([top_left, top_right, bottom_right, bottom_left])

def compute_corner_slope(corners):
    """
    Given an array of 4 corner points (x,y) from sorted,
    find the two topmost and two bottommost corners, order them left to right, and compute sign of their slopes.

    Args:
      corners: (4,2) array-like of pixel coords [(x0,y0),...,(x3,y3)] in image coordinates.

    Returns:
      sign of slope = sign((y_right_top - y_left_top) / (x_right_top - x_left_top) + (y_right_bottom - y_left_bottom) / (x_right_bottom - x_left_bottom))
    """
    pts = np.asarray(corners)

    # Top two corners (smallest y)
    top2_idx = np.argsort(pts[:,1])[:2]
    top2 = pts[top2_idx]
    left_top, right_top = top2[np.argsort(top2[:,0])]
    dx_top = right_top[0] - left_top[0]
    dy_top = right_top[1] - left_top[1]
    if dx_top == 0:
        slope_top = 0
    else :
        slope_top = dy_top / dx_top
    print('Slope top : ', slope_top)

    # Bottom two corners (largest y)
    bottom2_idx = np.argsort(pts[:,1])[-2:]
    bottom2 = pts[bottom2_idx]
    left_bot, right_bot = bottom2[np.argsort(bottom2[:,0])]
    dx_bot = right_bot[0] - left_bot[0]
    dy_bot = right_bot[1] - left_bot[1]
    if dx_bot == 0:
        slope_bot = 0
    else :
        slope_bot = dy_bot / dx_bot
    print('Slope bot : ', slope_bot)
    
    slope = slope_top + slope_bot
    print("Slope: ", np.sign(slope))

    if slope >= 0:
        return 1
    elif slope < 0:
        return -1

def quaternion2rotmat(quaternion):      
    """
    Compute the rotation matrix from a quaternion

    Inputs:
                quaternion: A list of 4 numbers [x, y, z, w] that represents the quaternion
    Outputs:
                R: A 3x3 numpy array that represents the rotation matrix of the quaternion
    """
    R = np.eye(3)
    x, y, z, w = quaternion
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    return R

def triangulation_preprocess(pixel, sensor_data):
    """""
    Triangulation preprocess function to compute the image vector and camera position in inertial frame

    Inputs:
            pixel: Pixel coordinates in the image, tuple of 2 numbers (x, y)
            sensor_data: Position of the drone in inertial frame, np array [x, y, z]
    Outputs:
            r: image vector in inertial frame ndarray of shape (3,)
            position: camera position in inertial frame ndarray of shape (3,) 
    """

    v_vector = np.array([pixel[0], pixel[1], FOCAL_LENGTH])      # image vector in camera frame             
    R_b2i = quaternion2rotmat([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]) 
    R_c2i = R_b2i @ R_C2B
    r = R_c2i @ v_vector                                         # image vector in inertial frame

    cam_offset_body = np.array([X_CAM, Y_CAM, Z_CAM])            # camera offset in body frame
    cam_offset_inertial = R_b2i @ cam_offset_body                # camera offset in inertial frame
    position = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]) + cam_offset_inertial # Position of the camera in the inertial frame
    print('Camera position : ', position)
    return r, position

def triangulation(t):
    """
    Triangulation function to compute the position of the corners and centroid of the gate in inertial frame

    """
    r = r_s_vectors[:, t, 0]
    s = r_s_vectors[:, t, 1]
    p = p_q_vectors[:, t, 0]
    q = p_q_vectors[:, t, 1]
    A = np.array([[np.dot(r, r), -np.dot(r, s)],
                [np.dot(s, r), -np.dot(s, s)]])
    b = np.array([np.dot((q - p), r), np.dot((q - p), s)])
    sol = np.linalg.solve(A,b)
    lambda_ = sol[0]
    mu = sol[1]
    f = p + lambda_ * r
    g = q + mu * s
    position = (f + g) / 2
    return position

def plane_normal(corners):
    """
    Compute the unit normal of the plane defined by 4 corners.

    Inputs :
                corners : Corners of the gate, list of four ndarrays of shape (3,).

    Outputs :
                n : Unit normal vector pointing in the halfspace of the drone, ndarray of shape (3,).
    """
    # Pick p0, p1, p2
    p0, p2, p3 = corners[0], corners[2], corners[3] # top left, bottom right, bottom left corners
    # Form two edge vectors
    v1 = p2 - p3
    v2 = p0 - p3
    # Cross product = normal (not yet unit)
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / norm

def normal_points(normal, centroid):
    """
    Compute the normal points to go through when passing the gate.

    Inputs:
                normal: unit normal vector of the plane defined by the 4 corners. This vector positive sign is always in the halfspace (delimited by gate plane) of the drone.
                centroid: centroid of the gate in inertial frame

    Outputs:
                np1: first normal point in inertial frame
                np2: second normal point in inertial frame
    """
    np1 = centroid + DISTANCE_FROM_GATE * normal
    np2 = centroid - DISTANCE_FROM_GATE * normal
    return np1, np2

def displacement(initial_position, yaw, move_radius = 0.5, deviation = 0, in_yaw_direction = 1):
    """
    Compute the displacement vector to move around the initial position in a circle.
        Inputs:
                 initial_position: position [x, y, z] in inertial frame
                 yaw: yaw angle in radians
                 move_radius: radius of the circle to move around the initial position
                 deviation: additional deviation to add to the displacement vector
                 in_yaw_direction: 1 to move in the yaw direction, 0 to move in x direction only
        Outputs:
                 command: output command [x, y, z, yaw] to be sent to drone
    """

    goal_x = initial_position[0] + move_radius * np.cos(yaw*in_yaw_direction) + deviation  # slightly displace from the current x position 
    goal_y = initial_position[1] + move_radius * np.sin(yaw*in_yaw_direction) + deviation  # slightly displace from the current y position 
    command = [goal_x, goal_y, initial_position[2], yaw] 
    return command

def visualize_gates(GATES_DATA):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot map plane
    # Map is 8x8m, origin at (0,0,0) lower right, x to left, y upward
    # We'll draw the square
    xx = [0, 8, 8, 0, 0]
    yy = [0, 0, 8, 8, 0]
    zz = [0]*5
    ax.plot(xx, yy, zz, color='gray', linestyle='--')

    # Plot gate features
    for key, data in GATES_DATA.items():
        centroid = data['centroid']
        corners = data['corners']
        normal_pts = data['normal points']

        # Plot centroid
        if centroid is not None:
            ax.scatter(*centroid, color='magenta', s=50, label=f"{key} centroid")

        # Plot corners
        for corner in corners:
            ax.scatter(*corner, color='blue', s=20)

        # Plot normal points
        for np_pt in normal_pts:
            ax.scatter(*np_pt, color='green', s=20)

    # Labels and legend
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    # Z limits based on data; assuming gates ~1m high
    ax.set_zlim(0, 2)

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()

# ----------------- Helper functions for second and third racing laps -----------------

def sort_wp_min_energy(wp, angle_penalty_weight=ANGLE_PENALTY):
    """
    Finds the most energy-efficient order to visit a series of gates, where:
    - Each gate has 2 normal points (entry/exit) that must be consecutive, but can flip.
    - The order of gates can be permuted.
    - The path starts at either normal point 1 or 2 (1st gate)
    - Energy = distance + angle penalties (to avoid sharp turns).
    
    Args:
        wp (list): Waypoints. List of 5 tuples, 1 (np.array(entry), np.array(exit)) for each gate.
        angle_penalty_weight (float): Weight applied to angle penalties relative to distance.

    Returns:
        best_path (list): List of np.array waypointspoints representing the optimal path.
        best_indices (list): List of indices (int) representing the updated order of waypoints.
        min_cost (float): Total energy cost (distance + angle penalties).
    """

    num_gates = len(wp) // 2  # Number of gates
    gate_indices = [ (2*i, 2*i+1) for i in range(num_gates) ]  # Gate groupings

    best_path = None
    best_indices = None
    min_cost = float('inf')

    for gate_order in permutations(gate_indices):
        for flip_config in product([False, True], repeat=num_gates):
            path = []
            indices = []
            for i, gate in enumerate(gate_order):
                idx1, idx2 = gate
                if flip_config[i]:
                    path.extend([wp[idx2], wp[idx1]])      # Flip order
                    indices.extend([idx2, idx1])
                else:
                    path.extend([wp[idx1], wp[idx2]])      # Normal order
                    indices.extend([idx1, idx2])

            # Ensure starting point is waypoint 0 or 1 (Gate 1's normal points)
            if indices[0] not in [0, 1]:
                continue

            # Compute distance + angle penalties
            dist = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
            angle_penalty = 0
            for i in range(1, len(path)-1):
                vec1 = path[i] - path[i-1]
                vec2 = path[i+1] - path[i]
                if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle_penalty += (1 - cos_theta)

            cost = dist + angle_penalty_weight * angle_penalty

            if cost < min_cost:
                min_cost = cost
                best_path = path
                best_indices = indices

    return best_path, best_indices, min_cost

def init_params(path_waypoints, t_f=T_FINAL):
    """
    Compute timing and limits for m-waypoint path.
    Inputs:
      - path_waypoints: list of m (x,y,z) tuples
      - t_f           : total time to traverse path [s]
    Returns:
      times      : 1 x m numpy array of time that each segment needs to be completed with
    """
    m = len(path_waypoints)
    segment_lengths = np.array([
        np.linalg.norm(np.array(path_waypoints[i+1]) - np.array(path_waypoints[i]))
        for i in range(m-1)
    ])
    total_length = np.sum(segment_lengths)

    # Segment times are proportional to their lengths
    segment_times = (segment_lengths / total_length) * t_f

    return segment_times

def compute_poly_matrix(t):
    """
    Build the 5x6 constraint matrix A_m(t) for a 5-constraint, 5th-order polynomial:
      [ pos; vel; acc; jerk; snap ] = A_m(t) @ [c5,c4,c3,c2,c1,c0]^T
    """
    return np.array([
        [  t**5,    t**4,   t**3,   t**2, t, 1],   # pos
        [5*t**4, 4*t**3, 3*t**2, 2*t,   1, 0],   # vel
        [20*t**3,12*t**2, 6*t,    2,     0, 0],   # acc
        [60*t**2,24*t,    6,      0,     0, 0],   # jerk
        [120*t,   24,     0,      0,     0, 0],   # snap
    ])

def compute_poly_coefficients(path_waypoints, times):
    """
    Solve for the minimum-jerk poly coefficients of each segment.
    Inputs:
      - path_waypoints: list of m (x,y,z) tuples
      - times         : 1xm array of waypoint times
    Returns:
      poly_coeffs: (6*(m-1))x3 array, stacking [c5...c0] vertically for each segment & horizontally for each dim
    """
    seg_times = np.diff(times)
    m = len(path_waypoints)
    n_segs = m - 1

    poly_coeffs = np.zeros((6*n_segs, 3))

    for dim in range(3):                                    # Compute for x, y, and z separately
        A = np.zeros((6*(n_segs), 6*(n_segs)))
        b = np.zeros(6*(n_segs))
        pos = np.array([p[dim] for p in path_waypoints])
        A_0 = compute_poly_matrix(0)                        # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

        row = 0
        for i in range(n_segs):
            pos_0, pos_f = pos[i], pos[i+1]                 #Starting and final positions of the segment
            v_0 = a_0 = v_f = a_f = 0                       # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
            A_f = compute_poly_matrix(seg_times[i])         # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
            if i == 0:                                      # First path segment
                # start pos, end pos, start vel, start acc
                A[row, i*6:(i+1)*6] = A_0[0]                #Initial position constraint
                b[row] = pos_0
                row += 1
                A[row, i*6:(i+1)*6] = A_f[0]                #Final position constraint
                b[row] = pos_f
                row += 1
                A[row, i*6:(i+1)*6] = A_0[1]                #Initial velocity constraint
                b[row] = v_0
                row += 1
                A[row, i*6:(i+1)*6] = A_0[2]                #Initial acceleration constraint
                b[row] = a_0
                row += 1
                # continuity vel/acc/jerk/snap between seg 0 & 1
                A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                b[row:row+4] = np.zeros(4)
                row += 4
            elif i < m-2:                                   # Intermediate path segments
                # intermediate: pos start/end + continuity
                A[row, i*6:(i+1)*6] = A_0[0]                #Initial position constraint
                b[row] = pos_0
                row += 1
                A[row, i*6:(i+1)*6] = A_f[0]                #Final position constraint
                b[row] = pos_f
                row += 1
                # continuity of velocity, acceleration, jerk and snap
                A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                b[row:row+4] = np.zeros(4)
                row += 4
            elif i == m-2:                                  #Final path segment
                # start pos, end pos, end vel, end acc
                A[row, i*6:(i+1)*6] = A_0[0]                #Initial position constraint
                b[row] = pos_0
                row += 1
                A[row, i*6:(i+1)*6] = A_f[0]                #Final position constraint
                b[row] = pos_f
                row += 1
                A[row, i*6:(i+1)*6] = A_f[1]                #Final velocity constraint
                b[row] = v_f
                row += 1
                A[row, i*6:(i+1)*6] = A_f[2]                #Final acceleration constraint
                b[row] = a_f
                row += 1

        # Solve for the polynomial coefficients for the dimension dim
        poly_coeffs[:,dim] = np.linalg.solve(A, b)   

    return poly_coeffs

def poly_setpoint_extraction(poly_coeffs, times):
    """
    From poly_coeffs and times, sample position, vel, accel at fine intervals.
    Also asserts speed/accel ≤ limits.
    Inputs:
      - poly_coeffs: (6*(m-1))x3 output of compute_poly_coefficients
      - times       : 1xm array
      - disc_steps  : samples per waypoint interval
      - vel_lim     : max speed
      - acc_lim     : max accel
      - tol         : small epsilon
    Returns:
      trajectory_setpoints : (Nx4) [x,y,z,yaw]
      time_setpoints       : (N,) sample times
    """
    m = len(times)
    total_samples = DISC_STEPS * m
    time_setpoints = np.linspace(times[0], times[-1], total_samples)

    x_vals = np.zeros((total_samples,1))
    y_vals = np.zeros((total_samples,1))
    z_vals = np.zeros((total_samples,1))
    v_vals = np.zeros((total_samples,1))
    a_vals = np.zeros((total_samples,1))

    coeffs_x = poly_coeffs[:,0]
    coeffs_y = poly_coeffs[:,1]
    coeffs_z = poly_coeffs[:,2]

    for i,t in enumerate(time_setpoints):
        # which segment
        idx = min(max(np.searchsorted(times, t)-1, 0), m-2)
        t_rel = t - times[idx]
        A = compute_poly_matrix(t_rel)

        x  = A[0] @ coeffs_x[idx*6:(idx+1)*6]
        y  = A[0] @ coeffs_y[idx*6:(idx+1)*6]
        z  = A[0] @ coeffs_z[idx*6:(idx+1)*6]
        vx = A[1] @ coeffs_x[idx*6:(idx+1)*6]
        vy = A[1] @ coeffs_y[idx*6:(idx+1)*6]
        vz = A[1] @ coeffs_z[idx*6:(idx+1)*6]
        ax = A[2] @ coeffs_x[idx*6:(idx+1)*6]
        ay = A[2] @ coeffs_y[idx*6:(idx+1)*6]
        az = A[2] @ coeffs_z[idx*6:(idx+1)*6]

        x_vals[i] = x
        y_vals[i] = y
        z_vals[i] = z
        v_vals[i] = np.linalg.norm([vx,vy,vz])
        a_vals[i] = np.linalg.norm([ax,ay,az])

    # check limits
    assert np.max(v_vals) <= VEL_LIM, "V_max exceeded"
    assert np.max(a_vals) <= ACC_LIM, "A_max exceeded"

    yaw_vals = np.zeros((total_samples,1))
    trajectory_setpoints = np.hstack([x_vals, y_vals, z_vals, yaw_vals])

    return trajectory_setpoints, time_setpoints