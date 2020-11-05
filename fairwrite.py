import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.dockarea import *
import serial
import pandas as pd
import os
import datetime as dt
import numpy as np
import logging
import sys
import pytz
import copy
import time
import utilities_helper as ut
import skinematics as skin
import pyqtgraph.exporters
from PIL import Image
import torch
import OS_CNN as oscnn

LETTER_VOID_SPACE = 20
LETTER_SIZE = 64
GRAVITY = 9.80665
ROLL_IDX = 0
CANVAS_HEIGHT = 400
CANVAS_LENGTH = 800
CANVAS_X_MIN = -80
CANVAS_X_MAX = 80
CANVAS_Y_MIN = -70
CANVAS_Y_MAX = 70
CANVAS_X_MULTIPLIER = CANVAS_LENGTH / (CANVAS_X_MAX - CANVAS_X_MIN)
CANVAS_Y_MULTIPLIER = -CANVAS_HEIGHT / (CANVAS_Y_MAX - CANVAS_Y_MIN)
LAST_IDX = -1
WINDOW_SIZE = 5.0  # seconds
# controls the gain per step in Madgwicks algorithm. The greater the value, the faster
# correction by earth magenetic field in stationary position but more unstable.
BETA_MADGWICK = 0.1
LOG_FORMAT = "[%(levelname)s:%(name)s - %(funcName)20s():%(lineno)s] %(message)s"
UNIX_EPOCH_naive = dt.datetime(1970, 1, 1, 0, 0)  # offset-naive datetime
UNIX_EPOCH_offset_aware = dt.datetime(1970, 1, 1, 0, 0, tzinfo=pytz.utc)  # offset-aware datetime
UNIX_EPOCH = UNIX_EPOCH_naive
TS_MULT_us = 1e6


def now_timestamp(ts_mult=TS_MULT_us, epoch=UNIX_EPOCH):
    return int((dt.datetime.utcnow() - epoch).total_seconds() * ts_mult)


def int2dt(ts, ts_mult=TS_MULT_us):
    return dt.datetime.utcfromtimestamp(float(ts) / ts_mult)


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        # PySide's QTime() initialiser fails miserably and dismisses args/kwargs
        # return [QTime().addMSecs(value).toString('mm:ss') for value in values]
        return [int2dt(value).strftime("%Hh:%Mm:%S.%f")[:-3] for value in values]


class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = QtCore.pyqtSignal()
    # error = QtCore.pyqtSignal(tuple)
    # result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)


class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    # def __init__(self, fn):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        # self.fn(*self.args, **self.kwargs)
        self.fn()
        self.signals.finished.emit()  # Done



class SensorApplication(QtGui.QApplication):

    def __init__(self, *args, **kwargs):
        super(SensorApplication, self).__init__(*args, **kwargs)

        # logger properties setup
        self.folder_name = "logs/"
        traces_name = self.folder_name + dt.datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss")
        self.cvs_name = traces_name + ".csv"
        self.logfile_name = traces_name + ".txt"
        os.makedirs(os.path.dirname(self.logfile_name), exist_ok=True)
        logging.basicConfig(filename=self.logfile_name, level=logging.DEBUG, datefmt="%H:%M:%S", format=LOG_FORMAT)
        self.logger1 = logging.getLogger('RawData')
        self.logger2 = logging.getLogger('ProgFlow')

        # Thread poll for getting serial line
        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(20)
        self.logger2.info("Multithreading with maximum %d threads" % self.thread_pool.maxThreadCount())

        # Serial properties setup
        self.serial_dev = serial.Serial()
        self.serial_dev.port = 'COM4'
        self.serial_dev.baudrate = 115200
        self.serial_dev.timeout = 2

        # graphic properties setup
        self.win = QtGui.QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1200, 650)
        self.win.setWindowTitle('IMU signal viewer')
        # Note that size arguments are only a suggestion; docks will still have to
        # fill the entire dock area and obey the limits of their internal widgets
        self.d1 = Dock("Accelerometer", size=(400, 400))
        self.d2 = Dock("Magnetometer", size=(400, 400))
        self.d3 = Dock("Gyroscope", size=(400, 400))
        self.d4 = Dock("LinearAcc", size=(800, 250))
        self.d5 = Dock("Position", size=(800, 400))
        self.d6 = Dock("Euler", size=(800, 250))
        self.d7 = Dock("Menu", size=(50, 50))
        self.d8 = Dock("Calibration", size=(400, 400))
        self.d9 = Dock("Graph", size=(CANVAS_LENGTH, CANVAS_HEIGHT))
        self.d10 = Dock("Word", size=(800, 250))
        self.d11 = Dock("Clasification", size=(400, 400))
        self.area.addDock(self.d4, 'left')  # place self.d6 at left edge of dock self.area
        self.area.addDock(self.d10, 'above', self.d4)
        self.area.addDock(self.d6, 'above', self.d10)
        self.area.addDock(self.d5, 'bottom', self.d6)  # place self.d9 bottom edge of self.d6
        self.area.addDock(self.d9, 'above', self.d5)
        self.area.addDock(self.d1, 'right')
        self.area.addDock(self.d3, 'bottom', self.d1)
        self.area.addDock(self.d8, 'bottom', self.d3)
        self.area.addDock(self.d11, 'above', self.d8)
        self.area.addDock(self.d2, 'above', self.d11)
        self.area.addDock(self.d7, 'bottom', self.d2)
        plot_acc = pg.PlotWidget(title="Accelerometer m/s^2")
        plot_acc.addLegend()
        plot_acc.setLabel('bottom', 'Time', units='s')
        plot_acc.setYRange(-60, 60)
        self.acc_x = plot_acc.plot(pen=pg.mkPen('r', width=2), name='x')
        self.acc_y = plot_acc.plot(pen=pg.mkPen('b', width=2), name='y')
        self.acc_z = plot_acc.plot(pen=pg.mkPen('g', width=2), name='z')
        self.d1.addWidget(plot_acc)
        plot_mag = pg.PlotWidget(title="Magnetometer uT")
        plot_mag.addLegend()
        plot_mag.setLabel('bottom', 'Time', units='s')
        plot_mag.setYRange(-90, 90)
        self.mag_x = plot_mag.plot(pen=pg.mkPen('r', width=2), name='x')
        self.mag_y = plot_mag.plot(pen=pg.mkPen('b', width=2), name='y')
        self.mag_z = plot_mag.plot(pen=pg.mkPen('g', width=2), name='z')
        self.d2.addWidget(plot_mag)
        plot_gyr = pg.PlotWidget(title="Gyroscope rad/s")
        plot_gyr.addLegend()
        plot_gyr.setLabel('bottom', 'Time', units='s')
        plot_gyr.setYRange(-1500, 1500)
        self.gyr_x = plot_gyr.plot(pen=pg.mkPen('r', width=2), name='x')
        self.gyr_y = plot_gyr.plot(pen=pg.mkPen('b', width=2), name='y')
        self.gyr_z = plot_gyr.plot(pen=pg.mkPen('g', width=2), name='z')
        self.d3.addWidget(plot_gyr)
        plot_lin = pg.PlotWidget(title="Linear Accelerometer m/s^2")
        plot_lin.addLegend()
        plot_lin.setLabel('bottom', 'Time', units='s')
        plot_lin.setYRange(-60, 60)
        self.lin_x = plot_lin.plot(pen=pg.mkPen('r', width=2), name='x')
        self.lin_y = plot_lin.plot(pen=pg.mkPen('b', width=2), name='y')
        self.lin_z = plot_lin.plot(pen=pg.mkPen('g', width=2), name='z')
        self.d4.addWidget(plot_lin)
        plot_eul = pg.PlotWidget(title="Euler degrees")
        plot_eul.addLegend()
        plot_eul.setLabel('bottom', 'Time', units='s')
        plot_eul.setYRange(-200, 200)
        self.eul_x = plot_eul.plot(pen=pg.mkPen('r', width=2), name='x')
        self.eul_y = plot_eul.plot(pen=pg.mkPen('b', width=2), name='y')
        self.eul_z = plot_eul.plot(pen=pg.mkPen('g', width=2), name='z')
        self.d6.addWidget(plot_eul)

        # canvas dock
        self.label = QtWidgets.QLabel()
        self.canvas = QtGui.QPixmap(800, 400)
        self.canvas.fill(QtCore.Qt.white)
        self.label.setPixmap(self.canvas)
        self.d9.addWidget(self.label)
        self.last_x, self.last_y = CANVAS_LENGTH/2, CANVAS_HEIGHT/2
        self.last_ang_x, self.last_ang_y = 0.0, 0.0

        # display word dock
        layout_word = pg.LayoutWidget()
        self.label_word = QtGui.QLabel()
        layout_word.addWidget(self.label_word, row=0, col=0)
        self.d10.addWidget(layout_word)

        # plot position dock
        self.plot_position_wg = pg.PlotWidget(title="Character Display")
        self.plot_position_wg.setYRange(-0.50, 0.50)
        self.plot_position_wg.setXRange(-0.50, 0.50)
        self.plot_position_wg.addLegend()
        self.position_coord = self.plot_position_wg.plot(pen=pg.mkPen('r', width=5))
        self.d5.addWidget(self.plot_position_wg)
    
        # calibration window dock
        layout_calibration = pg.LayoutWidget()
        label_calib = QtGui.QLabel("Calibration:")
        self.label_calib_acc = QtGui.QLabel("ACC: 0")
        self.label_calib_gyr = QtGui.QLabel("GYR: 0")
        self.label_calib_mag = QtGui.QLabel("MAG: 0")
        self.label_calib_sys = QtGui.QLabel("SYS: 0")
        label_quat = QtGui.QLabel("Quaternion:")
        label_eul = QtGui.QLabel("Euler:")
        self.label_quaternion = QtGui.QLabel("W: 0.0\tX: 0.0\tY: 0.0\tZ: 0.0")
        self.label_euler = QtGui.QLabel("X: 0.0\tY: 0.0\tZ: 0.0")
        layout_calibration.addWidget(label_calib, row=0, col=0)
        layout_calibration.addWidget(self.label_calib_acc, row=1, col=0)
        layout_calibration.addWidget(self.label_calib_gyr, row=2, col=0)
        layout_calibration.addWidget(self.label_calib_mag, row=3, col=0)
        layout_calibration.addWidget(self.label_calib_sys, row=4, col=0)
        layout_calibration.addWidget(label_quat, row=5, col=0)
        layout_calibration.addWidget(self.label_quaternion, row=5, col=1)
        layout_calibration.addWidget(label_eul, row=6, col=0)
        layout_calibration.addWidget(self.label_euler, row=6, col=1)
        self.d8.addWidget(layout_calibration)
        
        # classification window dock
        layout_letter = pg.LayoutWidget()
        self.label_letter = QtGui.QLabel("")
        self.label_letter.setFont(QtGui.QFont('Arial',50))
        layout_letter.addWidget(self.label_letter, row=0, col=0)
        self.d11.addWidget(layout_letter)

        # Menu window dock
        layout_buttons = pg.LayoutWidget()
        connect_b = QtGui.QPushButton("Connect")  # To start listening to Serial port
        self.record_b = QtGui.QPushButton("Record")  # To start Recording data
        self.stop_b = QtGui.QPushButton("Stop")  # To start Recording data
        self.disconnect_b = QtGui.QPushButton("Disconnect Device")  # To start Recording data
        self.change_name_b = QtGui.QPushButton("Set log name")  # To start Recording data
        recenter_b = QtGui.QPushButton("Re-Center")  # Re-center pointer position
        self.textbox = QtGui.QLineEdit()
        label_1 = QtGui.QLabel("""Orientation Mode: """)
        self.mode_box = QtGui.QComboBox()
        self.mode_box.addItems(["complementary_filter", "Madgwick", "sensor_ahrs"])
        label_2 = QtGui.QLabel("""Word Mode: """)
        self.word_box = QtGui.QComboBox()
        self.word_box.addItems(["OFF", "ON"])
        label_3 = QtGui.QLabel("""Word command: """)
        self.start_b = QtGui.QPushButton("Start")  # To start listening to Serial port
        self.next_b = QtGui.QPushButton("Next")  # To start Recording data
        self.end_b = QtGui.QPushButton("End")  # To start Recording data
        layout_buttons.addWidget(connect_b, row=0, col=0)
        layout_buttons.addWidget(self.record_b, row=0, col=1)
        layout_buttons.addWidget(self.stop_b, row=0, col=2)
        layout_buttons.addWidget(self.disconnect_b, row=0, col=3)
        layout_buttons.addWidget(self.change_name_b, row=1, col=0)
        layout_buttons.addWidget(self.textbox, row=1, col=1, colspan=3)
        layout_buttons.addWidget(label_1, row=2, col=0)
        layout_buttons.addWidget(self.mode_box, row=2, col=1, colspan=2)
        layout_buttons.addWidget(recenter_b, row=2, col=3)
        layout_buttons.addWidget(label_2, row=3, col=0)
        layout_buttons.addWidget(self.word_box, row=3, col=1, colspan=2)
        layout_buttons.addWidget(label_3, row=4, col=0)
        layout_buttons.addWidget(self.start_b, row=4, col=1)
        layout_buttons.addWidget(self.next_b, row=4, col=2)
        layout_buttons.addWidget(self.end_b, row=4, col=3)
        connect_b.pressed.connect(self.listen)
        self.record_b.pressed.connect(self.record)
        self.stop_b.pressed.connect(self.stop_recording)
        self.disconnect_b.pressed.connect(self.disconnect)
        self.change_name_b.pressed.connect(self.change_name)
        self.start_b.pressed.connect(self.start_letter)
        self.next_b.pressed.connect(self.next_letter)
        self.end_b.pressed.connect(self.end_letter)
        recenter_b.pressed.connect(self.recenter_pointer)
        self.mode_box.currentIndexChanged.connect(self.mode_change)
        self.word_box.currentIndexChanged.connect(self.word_mode_change)
        self.start_b.setEnabled(False)
        self.next_b.setEnabled(False)
        self.end_b.setEnabled(False)
        self.d7.addWidget(layout_buttons)

        self.mutex = QtCore.QMutex()  # mutex to control raw_data access
        self.mutex_line = QtCore.QMutex()  # mutex to control raw_line access
        self.raw_line = []  # This list will keep the incoming lines to process
        self.imu_struct = {"TIME": [],
                         "ELAPSED_SECONDS": [],
                         "PSTATE": [],
                         "acc_x": [],
                         "acc_y": [],
                         "acc_z": [],
                         "mag_x": [],
                         "mag_y": [],
                         "mag_z": [],
                         "gyr_x": [],
                         "gyr_y": [],
                         "gyr_z": [],
                         "lin_x": [],
                         "lin_y": [],
                         "lin_z": [],
                         "eul_x": [],
                         "eul_y": [],
                         "eul_z": [],
                         "qua_w": [],
                         "qua_x": [],
                         "qua_y": [],
                         "qua_z": []}
        # Prettier and better solution would be to include this values to
        # imu_struct but that would imply REFACTORING the code to calculate
        # the x & y position in the fill table thread instead of the plotting thread
        self.empty_xy_points = {"x": [], "y": []}
        self.canvas_xy_points = copy.deepcopy(self.empty_xy_points)
        ##########################################################################################
        #  The rate of receiving RAW data depends on how much data is being transmitted
        # ########################################################################################
        #  Data received            |   Possible Calculation Methods    |   speed
        #  --------------------------------------------------------------------------------
        #  acc + gyr + quat         |   comp. filter                    |   Hz = 33  period = 30ms
        ##########################################################################################
        #  Orientation Modes:
        #   complementary_filter    Not working correctlty, can be improved or removed
        #   Madgwick                Using Madgwick's Sensor Fusion Algorithm
        #   sensor_ahrs             Using the Orientation provided by the sensor
        ##########################################################################################
        self.mode = "sensor_ahrs"
        self.period = 0.035   # The period depends on the transmission rate from the finger wearable
        self.rate = 1/self.period
        # this is the structure where the las 50 samples are stored for displaying on the real-time
        # plots on screen.
        self.raw_data = copy.deepcopy(self.imu_struct)
        # empty structure, must be keep that way
        self.data_struct = copy.deepcopy(self.imu_struct)
        # This is the structure where the data is stored for storing the online data after completing
        # the finger-movement.
        self.to_save_data = copy.deepcopy(self.raw_data)
        
        # Display the GUI
        self.win.show()
        
        # Calibration from BNO055 IMU sensor
        self.accel_calibration = 0
        self.gyro_calibration = 0
        self.magnet_calibration = 0
        self.system_calibration = 0
        
        # PyQt timer
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_graph)
        
        #flags to control de state machine (see Thesis Document chapter 3.4)
        self.first_angle_measure = True
        self.first_line = True
        self.first_update = True
        self.first_graph_update = True
        self.stop_threads = False
        self.init_time = 0
        self.recording = False
        self.word_mode = False
        self.recording_word = False
        self.word_letter_number = 0
        # Madgwick variable
        # By Increasing the value of Beta(magnetic & acc correction to gyro) is faster but more unstable
        self.madgwick_ahrs = skin.imus.Madgwick(rate=self.rate, Beta=0.10)
        self.prediction_model = None
        self.load_model()
        self.word_prediction = ""

    def fill_table(self):
        """
            fill_table: This function is executed by the listener thread every 
                        time a new data from the finger-worn sensor is received.
        """

        current_time = dt.datetime.now()
        self.mutex.lock()
        self.mutex_line.lock()
        line = self.raw_line.pop(0)
        self.mutex_line.unlock()
        line = str(line, encoding="ascii")
        line = line[:len(line) - 2]  # don't add last \n character

        self.append_raw_data(line)
        self.calculate_attitude()
        self.calculate_linear_acc()

        if self.first_update:
            self.init_time = dt.datetime.now()
            elapsed_sec = 0.0
            self.first_update = False
        else:
            elapsed_sec = (current_time - self.init_time).total_seconds()

        self.raw_data["ELAPSED_SECONDS"].append(elapsed_sec)
        if elapsed_sec >= WINDOW_SIZE:
            self.pop_raw_data()

        if self.recording:
            self.append_to_save_data()

        self.mutex.unlock()

    def calculate_linear_acc(self):
        acc = self.get_raw_data("acc")
        if self.mode == "Madgwick" or self.mode == "sensor_ahrs":
            quat = self.get_raw_data("qua")
            linear_acc = ut.rotate_acceleration(acc, quat)
            if self.mode == "Madgwick":
                linear_acc = ut.switch_xy(linear_acc)
                linear_acc[1] = linear_acc[1]*-1
        elif self.mode == "complementary_filter":
            eul = self.get_raw_data("eul")
            eul = np.deg2rad(eul)
            acc = ut.switch_xz(acc)
            linear_acc = ut.transform_by_euler_angle(acc, eul)
            linear_acc = ut.switch_xz(linear_acc)

        # Remove gravity from z axis
        linear_acc[2] = linear_acc[2] - GRAVITY
        # Invert signal
        linear_acc[1] = linear_acc[1]*-1
        linear_acc[2] = linear_acc[2]*-1

        self.append_lin_data(linear_acc)

    def calculate_attitude(self):
        acc = self.get_raw_data("acc")
        gyr = self.get_raw_data("gyr")
        mag = self.get_raw_data("mag")

        # update madgwick's ahrs regardless of the mode
        acc_madg = acc/np.linalg.norm(acc)
        mag_madg = acc/np.linalg.norm(mag)
        gyr_madg = np.radians(gyr)
        self.madgwick_ahrs.Update(gyr_madg, acc_madg, mag_madg)
        quaternion_data = self.madgwick_ahrs.Quaternion
        euler_data = skin.quat.quat2deg(quaternion_data)

        if self.mode == "complementary_filter":
            acc = ut.switch_xz(acc)
            if self.first_update:
                euler_data = ut.complementary_filter_attitude(acc, gyr,
                                                              self.rate, np.zeros(3))
            else:
                attitude = self.get_raw_data("eul")
                attitude = np.deg2rad(attitude)
                euler_data = ut.complementary_filter_attitude(acc, gyr,
                                                              self.rate, attitude)
            quaternion_data = ut.to_quaternion_from_euler(euler_data[1],
                                                          euler_data[2],
                                                          euler_data[0])
            euler_data = np.rad2deg(euler_data)

        if self.mode != "sensor_ahrs":
            self.append_quat_data(quaternion_data)
            self.append_eul_data(euler_data)

    def get_raw_data(self, val, idx=LAST_IDX):
        if val == "acc":
            res = [self.raw_data["acc_x"][idx],
                   self.raw_data["acc_y"][idx],
                   self.raw_data["acc_z"][idx]]
        elif val == "mag":
            res = [self.raw_data["mag_x"][idx],
                   self.raw_data["mag_y"][idx],
                   self.raw_data["mag_z"][idx]]
        elif val == "gyr":
            res = [self.raw_data["gyr_x"][idx],
                   self.raw_data["gyr_y"][idx],
                   self.raw_data["gyr_z"][idx]]
        elif val == "eul":
            res = [self.raw_data["eul_x"][idx],
                   self.raw_data["eul_y"][idx],
                   self.raw_data["eul_z"][idx]]
        elif val == "qua":
            res = [self.raw_data["qua_w"][idx],
                   self.raw_data["qua_x"][idx],
                   self.raw_data["qua_y"][idx],
                   self.raw_data["qua_z"][idx]]
        elif val == "pen":
            res = self.raw_data["PSTATE"][idx]
        else:
            res = 0.0
        return res

    def append_raw_data(self, line):
        time_stamp = ut.get_sensor_value("Tms", line)
        pen_state = ut.get_sensor_value("St", line)
        calibration = ut.get_sensor_value("CALIB", line)
        linear_data = ut.get_sensor_value("LIN", line)
        accelerometer_data = ut.get_sensor_value("ACC", line)
        gyroscope_data = ut.get_sensor_value("GYR", line)
        magnetometer_data = ut.get_sensor_value("MAG", line)
        quaternion_data = ut.get_sensor_value("QUAT", line)

        if time_stamp:
            self.raw_data["TIME"].append(time_stamp[0])
        if pen_state:
            self.raw_data["PSTATE"].append(pen_state[0])
        if calibration:
            self.accel_calibration = calibration[0]
            self.gyro_calibration = calibration[1]
            self.magnet_calibration = calibration[2]
            self.system_calibration = calibration[3]
        if linear_data:
            self.append_lin_data(linear_data)
        if accelerometer_data:
            self.append_acc_data(accelerometer_data)
        if gyroscope_data:
            self.append_gyr_data(gyroscope_data)
        if magnetometer_data:
            self.append_mag_data(magnetometer_data)
        else:
            self.append_mag_data([0.0, 0.0, 0.0])  # delete this line if mag data always expected
        if quaternion_data and self.mode == "sensor_ahrs":
            euler_data = skin.quat.quat2deg(quaternion_data)
            self.append_quat_data(quaternion_data)
            self.append_eul_data(euler_data)

    def append_acc_data(self,data):
        self.raw_data["acc_x"].append(data[0])
        self.raw_data["acc_y"].append(data[1])
        self.raw_data["acc_z"].append(data[2])

    def append_gyr_data(self,data):
        self.raw_data["gyr_x"].append(data[0])
        self.raw_data["gyr_y"].append(data[1])
        self.raw_data["gyr_z"].append(data[2])

    def append_mag_data(self,data):
        self.raw_data["mag_x"].append(data[0])
        self.raw_data["mag_y"].append(data[1])
        self.raw_data["mag_z"].append(data[2])

    def append_lin_data(self,data):
        self.raw_data["lin_x"].append(data[0])
        self.raw_data["lin_y"].append(data[1])
        self.raw_data["lin_z"].append(data[2])

    def append_eul_data(self,data):
        self.raw_data["eul_x"].append(data[0])
        self.raw_data["eul_y"].append(data[1])
        self.raw_data["eul_z"].append(data[2])

    def append_quat_data(self,data):
        self.raw_data["qua_w"].append(data[0])
        self.raw_data["qua_x"].append(data[1])
        self.raw_data["qua_y"].append(data[2])
        self.raw_data["qua_z"].append(data[3])

    def pop_raw_data(self):
        for key in self.raw_data:
            self.raw_data[key].pop(0)

    def append_to_save_data(self):
        for key in self.to_save_data:
            self.to_save_data[key].append(self.raw_data[key][-1])

    def show_text(self, quat, euler):
        self.label_calib_acc.setText("ACC: %d" % self.accel_calibration)
        self.label_calib_gyr.setText("GYR: %d" % self.gyro_calibration)
        self.label_calib_mag.setText("MAG: %d" % self.magnet_calibration)
        self.label_calib_sys.setText("SYS: %d" % self.system_calibration)
        quat_text = "W: " + str(round(quat[0], 4)) + "  X:" + str(round(quat[1], 4)) + \
                    "  Y: " + str(round(quat[2], 4)) + "  Z:" + str(round(quat[3], 4))
        euler_text = "  X:" + str(round(euler[0], 2)) + "  Y: " + str(round(euler[1], 2)) + \
                     "  Z:" + str(round(euler[2], 2))
        self.label_quaternion.setText(quat_text)
        self.label_euler.setText(euler_text)

    def update_graph(self):
        self.mutex.lock()
        data = copy.deepcopy(self.raw_data)
        self.mutex.unlock()

        quat_data = np.array([data["qua_w"][-1], data["qua_x"][-1], data["qua_y"][-1], data["qua_z"][-1]])
        current_angle = np.array([data["eul_x"][-1], data["eul_y"][-1], data["eul_z"][-1]])
        self.show_text(quat_data, current_angle)
        if self.mode == "sensor_ahrs" or self.mode == "Madgwick":
            curr_ang_x = current_angle[1]
            curr_ang_y = current_angle[2]
        else:
            curr_ang_x = current_angle[2]
            curr_ang_y = current_angle[1]

        if self.first_angle_measure:
            self.first_angle_measure = False
        else:
            angle_x = (curr_ang_x - self.last_ang_x) * CANVAS_X_MULTIPLIER
            angle_y = (curr_ang_y - self.last_ang_y) * CANVAS_Y_MULTIPLIER
            pan_x = np.around(angle_x + self.last_x)
            pan_y = np.around(angle_y + self.last_y)
            # Limit to Canvas frame
            pan_x = CANVAS_LENGTH if pan_x > CANVAS_LENGTH else pan_x
            pan_x = 0 if pan_x < 0 else pan_x
            pan_y = CANVAS_HEIGHT if pan_y > CANVAS_HEIGHT else pan_y
            pan_y = 0 if pan_y < 0 else pan_y
            self.draw_on_canvas(pan_x, pan_y, data["PSTATE"][-1])

        self.last_ang_x = curr_ang_x
        self.last_ang_y = curr_ang_y

        self.acc_x.setData(x=data["ELAPSED_SECONDS"], y=data["acc_x"])
        self.acc_y.setData(x=data["ELAPSED_SECONDS"], y=data["acc_y"])
        self.acc_z.setData(x=data["ELAPSED_SECONDS"], y=data["acc_z"])
        self.gyr_x.setData(x=data["ELAPSED_SECONDS"], y=data["gyr_x"])
        self.gyr_y.setData(x=data["ELAPSED_SECONDS"], y=data["gyr_y"])
        self.gyr_z.setData(x=data["ELAPSED_SECONDS"], y=data["gyr_z"])
        self.lin_x.setData(x=data["ELAPSED_SECONDS"], y=data["lin_x"])
        self.lin_y.setData(x=data["ELAPSED_SECONDS"], y=data["lin_y"])
        self.lin_z.setData(x=data["ELAPSED_SECONDS"], y=data["lin_z"])
        self.eul_x.setData(x=data["ELAPSED_SECONDS"], y=data["eul_x"])
        self.eul_y.setData(x=data["ELAPSED_SECONDS"], y=data["eul_y"])
        self.eul_z.setData(x=data["ELAPSED_SECONDS"], y=data["eul_z"])
        self.mag_x.setData(x=data["ELAPSED_SECONDS"], y=data["mag_x"])
        self.mag_y.setData(x=data["ELAPSED_SECONDS"], y=data["mag_y"])
        self.mag_z.setData(x=data["ELAPSED_SECONDS"], y=data["mag_z"])

    def get_line(self):
        self.logger2.info("Starting to Listen")

        line = self.serial_dev.readline()
        while True:
            line = self.serial_dev.readline()
            if line and not self.stop_threads:
                self.mutex_line.lock()
                self.raw_line.append(line)
                self.mutex_line.unlock()
                self.create_table_filler()
            else:
                self.logger2.warning("Get line thread ended. stop_threads: %s", self.stop_threads)
                break

    def create_table_filler(self):
        worker = Worker(self.fill_table)  # Any other args, kwargs are passed to the run function
        worker.signals.finished.connect(self.thread_finished_table_filler)
        self.thread_pool.start(worker)

    def thread_finished_table_filler(self):
        if self.first_graph_update:
            self.logger2.info("Start Updating Graph")
            self.timer.start(50)  # timer expiration in ms
            self.first_graph_update = False

    def thread_finished_get_signal(self):
        if not self.stop_threads:
            self.logger2.warning("Serial Connection lost, finishing listener thread")
            self.close_serial_port()
            self.timer.stop()   # stop updating graph
            self.first_update = True
            self.first_line = True
            self.first_graph_update = True
            self.recording = False

    def mode_change(self, idx):
        self.mode = self.mode_box.currentText()
        if "Madgwick" == self.mode:
            self.madgwick_ahrs = skin.imus.Madgwick(rate=self.rate, Beta=0.10)
        self.recenter_pointer()
        self.logger2.info("Changed orientation mode to: %s", self.mode)

    def word_mode_change(self, idx):
        if not self.recording:
            if "ON" == self.word_box.currentText():
                self.word_mode = True
                self.clear_canvas()
                self.record_b.setEnabled(False)
                self.stop_b.setEnabled(False)
                self.start_b.setEnabled(True)
                self.next_b.setEnabled(True)
                self.end_b.setEnabled(True)
            else:
                self.word_mode = False
                self.record_b.setEnabled(True)
                self.stop_b.setEnabled(True)
                self.start_b.setEnabled(False)
                self.next_b.setEnabled(False)
                self.end_b.setEnabled(False)
            self.logger2.info("Changed word mode to: %s", self.word_box.currentText())
        else:
            self.logger2.info("Can't change word mode while recording")

    def listen(self):
        if not self.serial_dev.is_open:
            self.stop_threads = False
            self.open_serial_port()
            self.logger2.info(self.serial_dev)
            self.logger2.info("Start get_line thread")
            worker = Worker(self.get_line)  # Any other args, kwargs are passed to the run function
            worker.signals.finished.connect(self.thread_finished_get_signal)  # callback when thread is finished
            # Execute
            self.thread_pool.start(worker)
        else:
            self.logger2.info("Serial Port already open")

    def record(self):
        if self.serial_dev.is_open:
            if not self.recording:
                self.logger2.info("Recording Sample")
                self.clear_canvas()
                self.recording = True
                self.change_name_b.setEnabled(False)
                self.word_box.setEnabled(False)
                self.disconnect_b.setEnabled(False)
            else:
                self.logger2.info("Already Recording")
        else:
            self.logger2.info("Device not connected")

    def start_letter(self):
        if self.serial_dev.is_open:
            if not self.recording:
                self.logger2.info("Recording Sample word")
                self.recording = True
                self.change_name_b.setEnabled(False)
                self.word_box.setEnabled(False)
            self.recording_word = True
        else:
            self.logger2.info("Device not connected")

    def next_letter(self):
        if self.recording:
            self.logger2.info("Saving letter of a word")
            self.save_data_into_file(str(self.word_letter_number))
            self.plot_position(str(self.word_letter_number))
            self.save_word_image()
            self.word_stitching()
            self.predict_character()
            self.recording_word = False
            self.to_save_data = copy.deepcopy(self.data_struct)
            self.canvas_xy_points = copy.deepcopy(self.empty_xy_points)
            self.clear_canvas()
            self.word_letter_number = self.word_letter_number + 1
        else:
            self.logger2.info("Not Recording")

    def disconnect(self):
        if not self.recording:
            self.logger2.info("Attempting to disconnect device")
            self.disconnect_device()
        else:
            self.logger2.info("Can't Disconnect while recording")

    def save_xy_points(self, x, y):
        self.canvas_xy_points["x"].append(x)
        self.canvas_xy_points["y"].append(y)

    def draw_on_canvas(self, current_x, current_y, pen_down):
        if pen_down and self.recording:
            if (not self.word_mode) or (self.word_mode and self.recording_word):
                self.draw_line(self.last_x, self.last_y, current_x, current_y)
                self.save_xy_points(current_x, current_y)
            else:
                self.draw_move_cursor(self.last_x, self.last_y, current_x, current_y)
        else:
            self.draw_move_cursor(self.last_x, self.last_y, current_x, current_y)

        self.last_x = current_x
        self.last_y = current_y
        self.label.update()

    def draw_line(self, prev_x, prev_y, x, y):
        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen(QtCore.Qt.black,10,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(prev_x, prev_y, x, y)
        painter.end()

    def draw_move_cursor(self, prev_x, prev_y, x, y):
        pen = QtGui.QPen(QtCore.Qt.white,10,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin)
        painter = QtGui.QPainter(self.label.pixmap())
        painter.setPen(pen)
        painter.drawPoint(prev_x, prev_y)
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)
        painter.drawPoint(x, y)
        painter.end()

    def draw_erase_last_point(self):
        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen(QtCore.Qt.white,10,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPoint(round(self.last_x), round(self.last_y))
        painter.end()

    def predict_character(self):
        self.logger2.info("Enter Function")
        # get the data
        file_name = self.cvs_name
        if self.recording_word:
            file_name = self.cvs_name[:-4] + "_" + str(self.word_letter_number) + ".csv"
        self.logger2.info(file_name)
        # file_name = "a_0.csv"
        dataframe = pd.read_csv(file_name)

        self.logger2.info("Extracting features")
        features = ut.f_features(dataframe).transpose()
        input_val = list()
        input_val.append(features)
        input_val = np.array(input_val)
        input_val = np.float32(input_val)
        input_val = torch.from_numpy(input_val)
        input_val.requires_grad = False
        self.logger2.info(str(input_val.shape))

        # predict the value
        self.logger2.info("Make prediction")
        pred_val = self.prediction_model(input_val)
        pred_val = pred_val.detach().cpu().numpy()
        self.logger2.info(pred_val)
        pred_val = np.argmax(pred_val[0])
        self.logger2.info(pred_val)

        pred_char = ut.decode_number(pred_val)
        log_msg = "Predicted character: " + pred_char
        self.logger2.info(log_msg)

        if self.recording_word:
            self.word_prediction = self.word_prediction + pred_char
            self.label_letter.setText(self.word_prediction)
        else:
            self.label_letter.setText(pred_char)

        return pred_val

    def load_model(self):
        # file containing the model
        model_f_name_1D_CNN = "./model/1D_CNN_36_classes_model/AirFinger_4Best_model"
        # network parameters
        input_shape = 60
        n_class = 36
        columns = 2
        Max_kernel_size = 89
        paramenter_number_of_layer_list = [8 * 128 * columns, 5 * 128 * 256 + 2 * 256 * 128]
        receptive_field_shape = min(int(input_shape / 4), Max_kernel_size)
        layer_parameter_list = oscnn.generate_layer_parameter_list(1,
                                                                   receptive_field_shape,
                                                                   paramenter_number_of_layer_list,
                                                                   in_channel=int(columns))
        # build os-cnn net and load weight
        self.prediction_model = oscnn.OS_CNN(layer_parameter_list, n_class, False)
        self.prediction_model.load_state_dict(torch.load(model_f_name_1D_CNN))
        self.prediction_model.eval()  # Sets the module in evaluation mode

    def save_image(self):
        self.logger2.info("Saving image")
        f_name = self.cvs_name[:-4] + "_angle" + ".png"
        image = QtGui.QImage(self.label.pixmap())
        image.save(f_name, format="PNG")
        f_name = self.cvs_name[:-4] + "_position" + ".png"
        exporter_position_img = pg.exporters.ImageExporter(self.plot_position_wg.plotItem.getViewBox())
        exporter_position_img.export(f_name)

    # crop, resize & save
    def save_word_image(self):
        self.logger2.info("Saving word letter")
        if not self.canvas_xy_points['x']:
            self.draw_erase_last_point()
            image = QtGui.QImage(self.label.pixmap())
        else:
            xMax = max(self.canvas_xy_points["x"]) + LETTER_VOID_SPACE
            xMin = min(self.canvas_xy_points["x"]) - LETTER_VOID_SPACE
            yMax = max(self.canvas_xy_points["y"]) + LETTER_VOID_SPACE
            yMin = min(self.canvas_xy_points["y"]) - LETTER_VOID_SPACE

            xMax = CANVAS_LENGTH if xMax > CANVAS_LENGTH else xMax
            xMin = 0 if xMin < 0 else xMin
            yMax = CANVAS_HEIGHT if yMax > CANVAS_HEIGHT else yMax
            yMin = 0 if yMin < 0 else yMin

            image = QtGui.QImage(self.label.pixmap())
            image = image.copy(xMin, yMin, xMax - xMin, yMax - yMin)

        image = image.scaled(LETTER_SIZE, LETTER_SIZE, QtCore.Qt.IgnoreAspectRatio)

        file_name = self.cvs_name[:-4] + "_" + str(self.word_letter_number) + ".png"
        image.save(file_name, format="PNG")
        return

    def word_stitching(self):
        # Here stitch letter to form word, save & display
        length = self.word_letter_number + 1
        word_image = Image.new('L', (LETTER_SIZE * length, LETTER_SIZE))
        for i in range(length):
            file_name = self.cvs_name[:-4] + "_" + str(i) + ".png"
            image = Image.open(file_name)
            word_image.paste(im=image, box=(i * LETTER_SIZE, 0))

        file_word_name = self.cvs_name[:-4] + ".png"
        word_image.save(file_word_name)

        display_image = QtGui.QPixmap(file_word_name)
        self.label_word.setPixmap(display_image)

        return

    def clear_canvas(self):
        self.logger2.info("Clearing canvas")
        self.canvas.fill(QtCore.Qt.white)
        self.label.setPixmap(self.canvas)

    def stop_recording(self):
        if self.recording:
            self.logger2.info("Stop recording sample")
            self.recording = False
            self.save_data_into_file()
            self.plot_position()
            self.save_image()
            self.predict_character()
            self.to_save_data = copy.deepcopy(self.data_struct)
            self.canvas_xy_points = copy.deepcopy(self.empty_xy_points)
            self.change_name_b.setEnabled(True)
            self.word_box.setEnabled(True)
            self.disconnect_b.setEnabled(True)
        else:
            self.logger2.info("Not Recording")

    def end_letter(self):
        if self.recording:
            self.logger2.info("Stop recording Word")
            if self.recording_word:
                self.next_letter()
            self.word_letter_number = 0
            self.word_prediction = ""
            self.recording = False
            self.recording_word = False
            self.change_name_b.setEnabled(True)
            self.disconnect_b.setEnabled(True)
            self.word_box.setEnabled(True)
        else:
            self.logger2.info("Not Recording")

    def plot_position(self, name_complement=None):
        file_name = self.cvs_name
        if name_complement:
            file_name = self.cvs_name[:-4] + "_" + name_complement + ".csv"

        data = pd.read_csv(file_name)
        linear_acc = [data["lin_x"], data["lin_y"], data["lin_z"]]
        coordinates = ut.calculate_position(linear_acc, self.period)
        if coordinates:
            self.position_coord.setData(x=coordinates[0], y=coordinates[1])

    def close_connection(self):
        self.disconnect_device()
        if self.recording:
            self.save_data_into_file()

    def change_name(self):
        if not self.recording:
            self.cvs_name = self.folder_name + self.textbox.text() + ".csv"
        else:
            self.logger2.info("Cannot change file name while recording")

    def save_data_into_file(self, name_complement=None):
        self.logger2.info("Saving data in cvs file")
        df = pd.DataFrame(self.to_save_data)
        file_name = self.cvs_name
        if name_complement:
            file_name = self.cvs_name[:-4] + "_" + name_complement + ".csv"
        df.to_csv(file_name, index=None, header=True)

    def recenter_pointer(self):
        self.draw_erase_last_point()
        self.last_x, self.last_y = CANVAS_LENGTH/2, CANVAS_HEIGHT/2
        self.first_angle_measure = True

    def disconnect_device(self):
        self.logger2.info("Disconnecting Device")
        self.stop_threads = True
        self.recenter_pointer()
        time.sleep(1)

        if self.serial_dev.is_open:
            self.close_serial_port()

    def open_serial_port(self):
        self.logger2.info("Opening port %s", self.serial_dev.port)
        self.serial_dev.open()

    def close_serial_port(self):
        self.serial_dev.close()


if __name__ == '__main__':

    print("Starting Program")
    app = SensorApplication(sys.argv)
    try:
        print("Connected")
        app.exec_()
    except serial.SerialException as ser_e:
        app.logger2.info(ser_e)
        print("Serial connection lost")
    finally:
        app.close_connection()

    print("Program Terminated")
