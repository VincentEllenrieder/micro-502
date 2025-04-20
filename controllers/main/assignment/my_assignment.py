import numpy as np
import time
import cv2
import matplotlib.pyplot as plt # to erase for assignment


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
    "Started" : 0,
    "Search next gate": 0,
    "Triangulate": 0,
    "Go to gate" : 0,
    "Finish": 0
}

INITIAL_YAW = -np.pi/2
ACCEPTABLE_YAW = np.pi/16

PINK_THRESHOLD_HIGH = 1
PINK_THRESHOLD_LOW = 1

def get_command(sensor_data, camera_data, dt):

    # Drone has not taken off
    if MANEUVER["Started"] == 0 :

        print('in Started', sensor_data['z_global'])
        goal_z = 1.0
        if sensor_data['z_global'] < goal_z :
            control_command = [sensor_data['x_global'], sensor_data['y_global'], goal_z, sensor_data['yaw']]
        else:
            MANEUVER['Started'] = 1
            MANEUVER["Search next gate"] = 1
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        return control_command 
    
    # Search next gate
    elif MANEUVER["Search next gate"] == 1 :
        
        gate_found = False
        print('in Search next gate', sensor_data['yaw'])
        
        # If gate not found : 
        if gate_found == False :

            # Set initial searching yaw
            if (sensor_data['yaw'] > ACCEPTABLE_YAW + ACCEPTABLE_YAW) and (sensor_data['yaw'] < ACCEPTABLE_YAW - ACCEPTABLE_YAW):
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], INITIAL_YAW]

            # When initial yaw reached, scan area starting from initial_yaw until pink area seen
            else:
                image = image_processing(camera_data)
                
                raise Exception('testing camera data')
            

        else:
            MANEUVER['Search next gate'] = 0
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

        return control_command
        
    else:   
        print('in else')
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
        return control_command
    

def image_processing(camera_data):
    img = camera_data[:, :, [2, 1, 0]] # convert BGRA to RGB
    img_th = (img[:, :, 0] < 220    )
    plt.imsave('image_blue_channel.png', img[:, :, 2])

    return img




    # # Take off example
    # if sensor_data['z_global'] < 0.49:
    #     control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']] # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
    
    #     return control_command