import math
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import re
from euclid import Quaternion as Quartclid
from euclid import Vector3
from sklearn.preprocessing import MaxAbsScaler
from scipy.signal import resample

CANVAS_HEIGHT = 400
CANVAS_LENGTH = 800
CANVAS_X_MIN = -80
CANVAS_X_MAX = 80
CANVAS_Y_MIN = -70
CANVAS_Y_MAX = 70
CANVAS_X_MULTIPLIER = CANVAS_LENGTH / (CANVAS_X_MAX - CANVAS_X_MIN)
CANVAS_Y_MULTIPLIER = -CANVAS_HEIGHT / (CANVAS_Y_MAX - CANVAS_Y_MIN)
RESAMPLE_VAL = 60


######################################################################################
#                       Utilities
######################################################################################
def to_quaternion_from_euler(yaw, pitch, roll):  # yaw (Z), pitch (Y), roll (X)
    # print(yaw, " ", pitch, " ", roll)
    # Abbreviations for the various angular functions
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = (cy * cp * cr) + (sy * sp * sr)
    x = (cy * cp * sr) - (sy * sp * cr)
    y = (sy * cp * sr) + (cy * sp * cr)
    z = (sy * cp * cr) - (cy * sp * sr)
    # print([w,x,y,z])
    return [w, x, y, z]


def transpose(data_matrix, data_type=None):
    """ transpose matrix and convert to specified data type.

    parameters:
        data_matrix: list of list
        data_type: type name, example: float
    returns:
        iterator of [x,y,z] list
    """
    data_matrix = zip(*data_matrix)
    data_matrix = [map(data_type, seq) for seq in data_matrix] if data_type else map(list, data_matrix)
    return data_matrix


def get_sensor_value(type_value, str_line):
    if type_value == "QUAT":
        match = re.match(rf".*{type_value}:\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)", str_line)
    elif type_value == "Tms":
        match = re.match(rf"^(\d+)", str_line)
    elif type_value == "St":
        match = re.match(rf"^\d+\s(\d)", str_line)
    elif type_value == "CALIB":
        match = re.match(rf"^\d+\s\d+\s(\d)(\d)(\d)(\d)", str_line)
    else:
        match = re.match(rf".*{type_value}:\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)", str_line)

    if not match:
        result = None
    else:
        result = [float(i) for i in match.groups()]

    return result


######################################################################################
#					obtaining attitude functions
######################################################################################
def switch_xz(vector):
    tmp = vector[2]
    vector[2] = vector[0]
    vector[0] = tmp
    return vector


def switch_xy(vector):
    tmp = vector[1]
    vector[1] = vector[0]
    vector[0] = tmp
    return vector


def transform_by_euler_angle(accel, euler):
    """
    """
    heading, attitude, bank = euler[1], euler[2], euler[0]
    return Quartclid.new_rotate_euler(heading, attitude, bank) * Vector3(*accel)


def attitude_by_acceleration(acc_tup):
    """
    calculate current attitude using only acceleration

    return data:
    a list contain roll angle, pitch angle and yaw angle
    [roll, pitch, yaw]
    """
    resultant = lambda num_list: np.sqrt(sum(map(np.square, num_list)))

    # assume roll is 0
    roll_angle = lambda acc_x, acc_y, acc_z: 0
    pitch_angle = lambda acc_x, acc_y, acc_z: np.arctan2(abs(acc_z), abs(acc_x))
    yaw_angle = lambda acc_x, acc_y, acc_z: -np.arctan2(abs(acc_y),
                                                        resultant([acc_x, acc_z])) if acc_y > 0 else np.arctan2(
        abs(acc_y), resultant([acc_x, acc_z]))

    # return [roll_angle(*acc_tup), pitch_angle(*acc_tup), (yaw_angle(*acc_tup))] # if in position B
    return [roll_angle(*acc_tup), pitch_angle(*acc_tup), (np.pi - yaw_angle(*acc_tup))]  # if in position A


def complementary_filter(alpha, current_attitude, euler_angle_by_acceleromter, euler_angle_by_gyroscope):
    return alpha * (current_attitude + euler_angle_by_gyroscope) + (1 - alpha) * euler_angle_by_acceleromter


# for online data processing
def complementary_filter_attitude(acc, gyr, rate, prev_attitude=np.zeros(3)):
    tau = 0.2
    cpf_alpha = tau / (tau + 1 / rate)
    current_attitude = prev_attitude

    gyr_rad = np.radians(gyr)
    angle_by_gyr = gyr_rad / rate
    angle_by_acc = np.array(attitude_by_acceleration(acc))
    current_attitude = complementary_filter(cpf_alpha, current_attitude,
                                            angle_by_acc, angle_by_gyr)
    return current_attitude


######################################################################################
#                   acceleration rotation and gravity removal
######################################################################################
def rotate_acceleration(accel, quat):
    """ rotate acc vectory by quaternion

    parameters:
        acc(list/vector):	acceleration vector xyz
        quat(list):			quaternion by elements w,x,y,z in a list
    returns:
        float ang_x,ang_y, ang_z:  angles
    """
    quaternion = Quaternion(quat)
    return quaternion.rotate(accel)


######################################################################################
#                   movement detection
######################################################################################
def movement_detection(acc_group_by_axis):
    """
    :param acc_group_by_axis:
    :return:
    """
    detection = []
    acc_by_tuples = transpose(acc_group_by_axis)
    for acc in zip(acc_by_tuples):
        if np.absolute(acc[0][1]) > 0.5 or np.absolute(acc[0][2]) > 0.5:
            # if np.absolute(acc[0][0]) > 0.5 or np.absolute(acc[0][1]) > 0.5 or np.absolute(acc[0][2]) > 0.5:
            detection.append(1)
        else:
            detection.append(0)
    return detection


def find_dynamic_interval(movement_detection):
    """ find all dynamic stage interval

    parameters:
        movement_detection(list): list containg a 1 when movement was detected
    return:
        list with tuples(begin,end) with the indexes of movement_detection that
        mark the begin and end of movement detected, used for correction.
    """
    interval_list = []

    # get all index that value is not 0
    dynamic_time = np.nonzero(movement_detection)[0]

    # find continuous value
    last, begin = -1, -1
    for t in dynamic_time:
        if last == -1:
            last, begin = t, t
        elif t - last == 1:
            last = t
        else:
            interval_list.append((begin, last + 1))
            last = -1
    else:
        interval_list.append((begin, last + 1 if last + 1 != len(movement_detection) else last))

    return interval_list


def dynamic_error_compensation(data_list, movement_detection):
    """ compensate the error from dynamic stage to static stage
    :param data_list: the complete data points recorded
    :param movement_detection: has the same size as data_list.
           a number one is place when movement is detected at that
           data point, otherwise 0.
    :return: returns a list of the compensated velocity
    """

    data_list = np.array(data_list)
    interval_list = find_dynamic_interval(movement_detection)

    for beg, end in interval_list:
        total_diff = data_list[end] - data_list[end - 1]
        step_diff = total_diff / (end - beg + 1)
        # interpolation
        data_list[beg:end] += [step_diff * (i + 1) for i in range(end - beg)]
    return list(data_list)


######################################################################################
#                   velocity and position functions
######################################################################################
# todo add movement_detection to parameters as an array
def get_velocity(acc_group_by_axis, movement_detection, time=0.060):
    """ obtain velocity from acceleration.

    parameters:
        acc_group_by_axis(list of list): [[ax values],[ay values],[az values]]
        time:	specific rate time
    returns:
        velocity in a list of list by axis: [[vx values],[vy values],[vz values]]
    """

    velocity_group_by_axis = []

    for acc_list in acc_group_by_axis:
        vel = 0.0
        velocity_list = []

        for acc, movement in zip(acc_list, movement_detection):
            vel = vel + (acc * time) if movement == 1 else 0
            velocity_list.append(vel)

        velocity_group_by_axis.append(velocity_list)

    velocity_group_by_axis = [dynamic_error_compensation(velocity_list, movement_detection) for velocity_list in
                              velocity_group_by_axis]

    return velocity_group_by_axis


def calculate_displacement(velocity_group_by_axis, time=0.060):
    displacement_group_by_axis = []
    for velocity_list in velocity_group_by_axis:
        displacement_list = []
        disp = 0.0

        for vel in zip(velocity_list):
            disp = disp + (vel[0] * time)
            displacement_list.append(disp)

        displacement_group_by_axis.append(displacement_list)
    return displacement_group_by_axis


def calculate_position(linear_acc, period):
    movement = False
    movement_detected = movement_detection(linear_acc)
    for i in movement_detected:
        if i == 1:
            movement = True
            break

    if not movement:
        return None
    ################################################################
    #    Calculate velocity and displacement
    ################################################################
    velocity = get_velocity(linear_acc, movement_detected, period)
    displacement = calculate_displacement(velocity)
    ################################################################
    #    Get coordinates
    ################################################################
    x_coordinate = [x * -1 for x in displacement[1]]
    y_coordinate = [y * -1 for y in displacement[2]]
    return [x_coordinate, y_coordinate]


######################################################################################
#                   Feature extraction
######################################################################################
def f_features(dataframe):
    # Data used to get features
    eul = dataframe.iloc[:, 15:18].values.transpose()

    # calculate features
    xy = get_xy_canvas(eul)

    # Add features to feature list
    features = [xy[0], xy[1]]
    features = np.array(features).transpose()

    # Feature scaling maximum absolute value
    ma = MaxAbsScaler()
    features = ma.fit_transform(features)

    # Resampling to 60 (Median lenght in database)
    features = resample(features, RESAMPLE_VAL)

    return features


# Function for adaptation
def get_xy_canvas(data):
    # data : (3,n) np array orientation angles
    # return: (2,n) np array with x & y points
    last_x, last_y = 0, 0
    last_ang_x, last_ang_y = 0, 0
    x = []
    y = []
    for i in range(len(data[0])):
        curr_ang_x = data[1][i]
        curr_ang_y = data[2][i]

        angle_x = (curr_ang_x - last_ang_x) * CANVAS_X_MULTIPLIER
        angle_y = (last_ang_y - curr_ang_y) * CANVAS_Y_MULTIPLIER
        pan_x = np.around(angle_x + last_x)
        pan_y = np.around(angle_y + last_y)
        last_ang_x, last_ang_y = curr_ang_x, curr_ang_y
        last_x, last_y = pan_x, pan_y
        x.append(pan_x)
        y.append(pan_y)
    return np.array([x, y])


def decode_number(number):
    switcher = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
                15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
                22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S',
                29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}
    return switcher.get(number, str(number))

