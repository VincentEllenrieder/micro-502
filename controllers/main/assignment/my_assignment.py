import numpy as np
import time
import cv2
from lib.simple_pid import PID  # Assuming your PID class is here
import matplotlib.pyplot as plt # to erase for assignment
import time



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
    # -- Start race ---
    "Race" : 0,
    # --- Finish ---
    "Finish": 0
}

# Take off height
TAKEOFF_Z = 1.0

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

# Triangulation parameters
N_IMAGES = 2                                                            # Number of images to take for triangulation
images_taken = 0                                                        # Number of images taken for triangulation
ALIGNMENT_MARGIN = 15                                                   # Margin to consider drone aligned with gate centroid
CENTROID_YAW_PID = PID(Kp=0.005, Ki=0.0003, Kd=0.001, output_limits=(-np.pi/20, np.pi/20))   # PID controller for centroid alignment
CENTROID_YAW_PID.setpoint = 0                                           # Alignment in Y should be at center of image     
R_C2B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])                   # Rotation matrix from camera frame to body frame   
DELAY = 0.5                                                             # Time delay between 2 images taken [s]
current_time = 0.0                                                      # Current time for time delay
aligned = False                                                         # Flag to check if drone has aligned with gate centroid
deviated = False                                                        # Flag to check if drone has deviated for second image     
r_s_vectors = np.zeros((3, N_IMAGES))                                   # Centroid image vectors in inertial frame  
p_q_vectors = np.zeros((3, N_IMAGES))                                   # Camera positions in inertial frame  

# Camera parameters
FOCAL_LENGTH = 161.013922282    # Focal length of the camera in pixels
X_CAM = 0.030                   # x position of the camera relative to the body frame in meters
Y_CAM = 0.000                   # y position of the camera relative to the body frame in meters
Z_CAM = 0.010                   # z position of the camera relative to the body frame in meters

# Thresholds for pink gate detection
RED_BLUE_THRESHOLD_HIGH = 230   # 190 < R, B < 230
RED_BLUE_THRESHOLD_LOW = 190 
GREEN_THRESHOLD = 180           # G < 180 

# Gates data
gates_found = 0
gates = []

def get_command(sensor_data, camera_data, dt):
    
    global MANEUVER, images_taken, gates_found, gates, aligned, deviated, r_s_vectors, p_q_vectors, current_time

    # Drone has not taken off
    if MANEUVER["Started"] == 0 :
        print('in Started', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])

        if sensor_data['z_global'] < TAKEOFF_Z :
            control_command = [sensor_data['x_global'], sensor_data['y_global'], TAKEOFF_Z, sensor_data['yaw']]
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
               (TAKEOFF_Z - POS_MARGIN < sensor_data['z_global'] < TAKEOFF_Z + POS_MARGIN) :
                print("At scan location")
                MANEUVER["Go to scan"] = 0
                MANEUVER["Vision"] = 1
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command 

            # Go to scan location if not there yet
            else :
                print("Not at scan location")
                control_command = [scan_location["SCAN_X"], scan_location["SCAN_Y"], TAKEOFF_Z, scan_location["SCAN_YAW"]]
                return control_command
            
        elif MANEUVER["Vision"] == 1:
            print('in Vision')
            
            while images_taken < N_IMAGES:
                
                centroid = get_centroid(camera_data)
                print("Centroid: ", centroid)

                # If no centroid found (orthogonal alignment with gate), displace
                if centroid == None:
                    print("No centroid found")
                    control_command = displacement([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']], sensor_data['yaw'])
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

                    deviation = displacement(p_q_vectors[:, 0], sensor_data['yaw'], deviation=0, in_yaw_direction=0) # Displace in direction of yaw when aligned to centroid
                    if not ((deviation[0] - POS_MARGIN < sensor_data['x_global'] < deviation[0] + POS_MARGIN) and \
                            (deviation[1] - POS_MARGIN < sensor_data['y_global'] < deviation[1] + POS_MARGIN)) and (not deviated) :
                        print("Deviating")
                        control_command = deviation
                        return control_command
                    deviated = True
                    # if current_time == 0.0: # if timer not started yet, start it 
                    #     current_time = time.time()

                    # print(time.time() - current_time)
                    # if time.time() - current_time < DELAY: # move for DELAY seconds before taking the next image
                    #     print("Deviating")
                    #     control_command = displacement([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']], sensor_data['yaw'], deviation = 0) 
                    #     return control_command

                v_vector = (np.array([centroid[0], centroid[1], FOCAL_LENGTH]))                
                R_b2i = quaternion2rotmat([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]) 
                R_c2i = R_b2i @ R_C2B
                r_s_vectors[:, images_taken] = R_c2i @ v_vector # Rotate the camera vector to the inertial frame

                cam_offset_body = np.array([X_CAM, Y_CAM, Z_CAM])
                cam_offset_inertial = R_b2i @ cam_offset_body
                position = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]) + cam_offset_inertial # Position of the camera in the inertial frame
                p_q_vectors[:, images_taken] = position 
                print('Camera position : ', position)
            
                images_taken += 1
                print("Image ", images_taken, " taken")
        

            r = r_s_vectors[:, 0]
            s = r_s_vectors[:, 1]
            p = p_q_vectors[:, 0]
            q = p_q_vectors[:, 1]
            A = np.array([[np.dot(r, r), -np.dot(r, s)],
                          [np.dot(s, r), -np.dot(s, s)]])
            b = np.array([np.dot((q - p), r), np.dot((q - p), s)])
            sol = np.linalg.solve(A,b)
            lambda_ = sol[0]
            mu = sol[1]
            print("Line coefficients: ", lambda_, mu)
            f = p + lambda_ * r
            g = q + mu * s
            gate_position = (f + g) / 2
            print("Gate position: ", gate_position)
            gates.append(gate_position) 

            deviated = False
            aligned = False
            r_s_vectors = np.zeros((3, N_IMAGES))                                   
            p_q_vectors = np.zeros((3, N_IMAGES))                                   
            MANEUVER["Vision"] = 0
            MANEUVER["Search next gate"] = 0
            MANEUVER["Go to gate"] = 1
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command 
    
    elif MANEUVER["Go to gate"] == 1:
        print('in Go to gate', sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'])
        
        goal = displacement([gates[gates_found][0], gates[gates_found][1], sensor_data['z_global']], sensor_data['yaw'], deviation=0) # Displace until slightly after gate position to go through the gate
        print("Goal: ", goal)
        if (goal[0] - POS_MARGIN < sensor_data['x_global'] < goal[0] + POS_MARGIN) and \
           (goal[1] - POS_MARGIN < sensor_data['y_global'] < goal[1] + POS_MARGIN) and \
           (goal[2] - POS_MARGIN < sensor_data['z_global'] < goal[2] + POS_MARGIN) and \
           (goal[3] - YAW_MARGIN < sensor_data['yaw'] < goal[3] + YAW_MARGIN) :
            print("At gate position")
            gates_found += 1
            MANEUVER["Go to gate"] = 0
            MANEUVER["Search next gate"] = 1
            if gates_found == 5:
                raise Exception('testing code')
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
            
        else:
            print("Moving to gate position")
            control_command = goal
            return control_command

    else:   
        print('in else')
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
        return control_command
      

def image_processing(camera_data):
    img = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2RGB) # convert BGRA to RGB
    plt.imsave('image_analysis/original_image.png', img[:, :, :])
    # Mask the image to only keep the pink area
    img_filtered = np.zeros(img.shape[:2], dtype=np.uint8)
    condition = (img[:, :, 0] > RED_BLUE_THRESHOLD_LOW) & (img[:, :, 0] < RED_BLUE_THRESHOLD_HIGH) & \
                (img[:, :, 2] > RED_BLUE_THRESHOLD_LOW) & (img[:, :, 2] < RED_BLUE_THRESHOLD_HIGH) & \
                (img[:, :, 1] < GREEN_THRESHOLD)
    img_filtered[condition] = 255
    plt.imsave('image_analysis/thresholding image.png', img_filtered[:, :], cmap='gray')
    return img_filtered

def get_centroid(camera_data):
    image_filtered = image_processing(camera_data)
    image_contours = np.zeros((*np.shape(image_filtered), 3), dtype=np.uint8) 
    contours, _ = cv2.findContours(image_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Find contours in the image
    #print('contours \n :', contours, ', len :', len(contours), ', shape :', np.shape(contours), ', type :', type(contours)) 
    
    # If sight orthogonal to gate's normal vector (no countours), signal to displace 
    if len(contours) == 0:
        print("No contours found")
        return None
    
    # If one contour found, directly compute its centroid
    elif len(contours) == 1:
        contour = contours[0]

        cv2.drawContours(image_contours, [contour], 0, (0, 255, 0), 1)          

        cX, cY = compute_centroid(contour)                                      
        centroid = cX - 150, cY - 150       # Shift centroid to center of image

        image_contours[cY, cX, 0] = 255     # Set the centroid pixel to red
        plt.imsave('image_analysis/gate_center_single_contour_image' + str(images_taken+1) + '.png', image_contours[:, :, :])

        return centroid                                                           

    # If multiple contours found, only take the one with rightmost centroid
    elif len(contours) > 1:

        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)

        rightmost = 0
        cX, cY = 0, 0
        for contour in contours:
            x, y = compute_centroid(contour)
            if x > rightmost:
                rightmost = x                   # Take the rightmost centroid
                centroid = x - 150, y - 150     # Shift centroid to center of image

        image_contours[cY, cX, 0] = 255         # Set the centroid pixel to red
        plt.imsave('image_analysis/gate_center_multiple_contours' + str(images_taken+1) + '.png', image_contours[:, :, :])

        return centroid

def compute_centroid(contours):
    # Compute the centroid of the given contour
    M = cv2.moments(contours)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        print("Error computing centroid")
    return cX, cY

def quaternion2rotmat(quaternion):
        
        R = np.eye(3)
        
        # Inputs:
        #           quaternion: A list of 4 numbers [x, y, z, w] that represents the quaternion
        # Outputs:
        #           R: A 3x3 numpy array that represents the rotation matrix of the quaternion

        x, y, z, w = quaternion
        
        R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
        return R

def displacement(initial_position, yaw, move_radius = 0.5, deviation = 0.2, in_yaw_direction = 1):
    goal_x = initial_position[0] + move_radius * np.cos(yaw*in_yaw_direction) + deviation * (2*np.random.rand() - 1) # slightly displace from the current x position 
    goal_y = initial_position[1] + move_radius * np.sin(yaw*in_yaw_direction) + deviation * (2*np.random.rand() - 1) # slightly displace from the current y position 
    command = [goal_x, goal_y, initial_position[2], yaw] 
    return command

# def deviate_z(sensor_data, deviation = 0.2):
#     goal_z = sensor_data['z_global'] + deviation # slightly deviate from the current z position
#     command = [sensor_data['x_global'], sensor_data['y_global'], goal_z, sensor_data['yaw']]
#     return command

    # # Take off example
    # if sensor_data['z_global'] < 0.49:
    #     control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']] # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
    
    #     return control_command