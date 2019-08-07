
try:  # Import library
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing GPIO! You need superuser privileges to use this")
from _thread import start_new_thread
from time import sleep

import cv2
import numpy as np
import os
import basicnavigation
import localization

__author__ = 'David Arnaiz'


"""
Robot IO control module:

This module defines the basic functions to control, set or read the sensors or actuators of the robot.

    -)Motor control: the robot is based on the L298N motor driver module. This module sets the speed and direction in
        which the motors spin. It takes one PWM and two regular pins for each motor.
        The robot speed is controlled with two motors and one caster wheel (differential model). The linear speed is set
        by the sum of the speed of the motors and the angular speed is set by the difference of speed between the two
        motors. The angular speed of the motors is corrected with the data collected by the encoders as the input
        voltage for the motors is unknown.
    -)Odometry based pose estimation: The data from the encoders is processed and the position of the robot is estimated
        based on the angular speed of the motors.
"""


# Variable definition

# Pin definition
# Right motor
pwmpina = 37
inpa1 = 35
inpa2 = 33
enca = 31  # Encoder of the right wheel
# Left motor
pwmpinb = 40
inpb1 = 38
inpb2 = 36
encb = 32  # Encoder of the left wheel
# Other pins
low_voltage = 16  # Low voltage detector input
shutdown_button = 18  # Shutdown button for the robot

# Robot parameters
r = 2.15  # Wheel radius in cm
l = 15  # Distance between motors in cm
marks = 12  # Number of marks for the encoder in the wheel
counta = 0  # Marks count for the right motor
countb = 0  # Marks count for the left motor
maxspeed = 140  # max linear speed in cm/s

# Initialization
back1 = False
back2 = False
error = 0
t_error = 0
v_correction = 0
strait = False

# PWM speeds
lright_speed = 0
lleft_speed = 0

# pose estimation coordinates (defined as global so the main thread can read them)
x_est = 0
y_est = 0
theta_est = 0
# Target poses
t_phi = 0  # target angle
t_y = 0  # initial target y coordinate
t_x = 0  # initial target x coordinate
a = 0
b = 0
# Kalman filter
q = np.matrix([[0.317, 0, 0], [0, 0.317, 0], [0, 0, 0.0056]])  # pose covariance
ppk = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # initial covariance matrix
var_v = 4207.459436  # vision variance
# camera measurements
xm = 6
ym = -2
zm = 13
# camera model
matrix, rms, distortion, ret = localization.read_camera_model()

#  Interrupts callbacks


def right_encoder(channel):  # Callback function for the interrupt for the encoders
    global counta
    if back1:  # If the motor is spining backward take one off
        counta -= 1
    else:  # if the motor goes forward add one
        counta += 1


def left_encoder(channel):
    global countb
    if back2:
        countb -= 1
    else:
        countb += 1


def shutdown(channel):  # Ends the robot
    if channel == shutdown_button:
        sleep(5)
        if GPIO.input(shutdown_button):
            return  # button not pressed
    elif channel == low_voltage:
        sleep(1)
        if not GPIO.input(low_voltage):
            return  # low voltage not detected
    global control
    control = False
    while not done:
        sleep(0.1)
    end_gpio()
    os.system("sudo shutdown -h now")

"""
Setup the GPIO pins for use
"""

GPIO.setmode(GPIO.BOARD)  # Set the mode for the pins

# Inputs
GPIO.setup(enca, GPIO.IN)
GPIO.setup(encb, GPIO.IN)
GPIO.setup(low_voltage, GPIO.IN)
GPIO.setup(shutdown_button, GPIO.IN)

# Outputs
GPIO.setup(inpa1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(inpa2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(inpb1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(inpb2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(pwmpina, GPIO.OUT)
GPIO.setup(pwmpinb, GPIO.OUT)

# PWM
pwma = GPIO.PWM(pwmpina, 500)  # frequency=500Hz
pwmb = GPIO.PWM(pwmpinb, 500)  # frequency=500Hz
pwma.start(0)
pwmb.start(0)

# Interrupts
"""
There will be 4 interrupts in total.
    -) right encoder: This interrupt counts the number of changes detected by the right encoder to keep track of the
        robot movements.
    -) left encoder: This interrupt counts the number of changes detected by the left encoder to keep track of the
       robot movements.
    -) Low voltage detector: This interrupt ends the operation of the robot when the the source voltage gets below
       the threshold. It just detects the change from Low (over the threshold) to High (below the threshold) and sets a
       debounce time of 5 s to avoid noise.
    -) Shutdown button: This interrupt ends the robot's operation safely and shutdown the Raspberry Pi and sets a
       debounce time of 5 s to avoid noise.
"""
# encoders
GPIO.add_event_detect(enca, GPIO.BOTH, callback=right_encoder)
GPIO.add_event_detect(encb, GPIO.BOTH, callback=left_encoder)
# Rest
# Only consider the shutdown imputs if they are active for 5 seconds
GPIO.add_event_detect(low_voltage, GPIO.RISING, callback=shutdown, bouncetime=5000)
GPIO.add_event_detect(shutdown_button, GPIO.FALLING, callback=shutdown, bouncetime=50000)

# camera set up
cap = cv2.VideoCapture(0)  # 0 is to select the main camera
# Just to capture video
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
out2 = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))


def end_gpio():  # Ends the GPIO pins cleanly
    """
    This function takes no inputs and generate no outputs.
    When the GPIO pins are not needed any more this function ends them cleanly, so the motors don't keep turning or the0
    encoders interrupts keep detecting edges. It's just called at the end when the battery power gets bellow the
    threshold.
    """
    pwma.stop()
    pwmb.stop()
    GPIO.remove_event_detect(enca)
    GPIO.remove_event_detect(encb)
    GPIO.cleanup()


def change_motors(right_speed, left_speed):  # change the PWM assigned to the motors
    """
    :Input --> right_speed: PWM value for the right motor.
    :Input --> left_speed: PWM value for the left motor.
    :Global input/output --> lright_speed: Previous PWM value for the right motor.
    :Global input/output --> lleft_speed: Previous PWM value for the left motor
    :Global output --> back1: sets that the right wheel is going backwards.
    :Global output --> back2: sets that the left wheel is going backwards.
    This function also uses the global variables 'back1' and 'back2' (just written by this function) so that the values
    change as soon as the motor direction changes to avoid delays.
    """
    global lright_speed, lleft_speed, back1, back2

    # Set right motor
    if not right_speed == lright_speed:  # only if the speed has changed
        lright_speed = right_speed
        if right_speed >= 100:
            GPIO.output(inpa1, True)
            GPIO.output(inpa2, False)
            pwma.ChangeDutyCycle(100)
            back1 = False
        elif right_speed >= 10:
            GPIO.output(inpa1, True)
            GPIO.output(inpa2, False)
            pwma.ChangeDutyCycle(right_speed)
            back1 = False
        elif right_speed < -100:
            GPIO.output(inpa1, False)
            GPIO.output(inpa2, True)
            pwma.ChangeDutyCycle(100)
            back1 = True
        elif right_speed <= -10:
            GPIO.output(inpa1, False)
            GPIO.output(inpa2, True)
            pwma.ChangeDutyCycle(-right_speed)
            back1 = True
        else:
            GPIO.output(inpa1, False)
            GPIO.output(inpa2, False)
            pwma.ChangeDutyCycle(0)

    # Set left motor
    if not lleft_speed == left_speed:  # Only if it has changed
        lleft_speed = left_speed
        if left_speed >= 100:
            GPIO.output(inpb1, True)
            GPIO.output(inpb2, False)
            pwmb.ChangeDutyCycle(100)
            back2 = False
        elif left_speed >= 10:
            GPIO.output(inpb1, True)
            GPIO.output(inpb2, False)
            pwmb.ChangeDutyCycle(left_speed)
            back2 = False
        elif left_speed <= -100:
            GPIO.output(inpb1, False)
            GPIO.output(inpb2, True)
            pwmb.ChangeDutyCycle(100)
            back2 = True
        elif left_speed <= -10:
            GPIO.output(inpb1, False)
            GPIO.output(inpb2, True)
            pwmb.ChangeDutyCycle(-left_speed)
            back2 = True
        else:
            GPIO.output(inpb1, False)
            GPIO.output(inpb2, False)
            pwmb.ChangeDutyCycle(0)


def target_position():  # obtains point of the target line closer to the actual pose
    """
    This point is the intersection between the perpendicular line to the target line through the robot's pose and the
    target line set every time the robt turns.
    """
    if a == 0:
        t_pose = np.matrix([[x_est], [y_est], [0], [1]])
    else:
        d = a*y_est - x_est  # constant of the perpendicular line
        target_x = (d-b)/(a-(1/a))
        target_y = a*target_x+b
        # robot pose
        t_pose = np.matrix([[target_x], [target_y], [0], [1]])
    t_pose =  np.matrix([[x_est], [y_est], [0], [1]])
    return t_pose


def change_speed(tturn, speed_angle, output, pos_x, pos_y):
    global back1, back2, strait, error, nact, t_phi, t_error, a, b, v_correction, t_x, t_y

    if tturn:  # if the turning mode is on
        strait = False
        lact = nact
        # Stop robot
        vela = velb = 0
        t_x = pos_x
        t_y = pos_y
        change_motors(vela, velb)
        # wait for actualization
        while nact == lact:
            sleep(0.1)
        tt_phi = t_phi + speed_angle
        try:
            # Normalise target angle
            speed_angle += t_phi - theta_est
            if speed_angle > np.pi:
                speed_angle -= 2*np.pi
            elif speed_angle < -np.pi:
                speed_angle += 2*np.pi
            t_phi = tt_phi
            if t_phi > np.pi:
                t_phi -= 2*np.pi
            elif t_phi < -np.pi:
                t_phi += 2*np.pi
            # obtain the target line
            if abs(t_phi) == np.pi/2:
                a = 0
                b = 0
            else:
                a = np.tan(t_phi)
                b = pos_y - a*pos_x
            t_anglea = int(round(speed_angle*((marks*l)/(2*np.pi*r))))  # target for right motor (positive )
            t_angleb = int(round(-speed_angle*((marks*l)/(2*np.pi*r))))  # target fot left motor
            if 0 <= speed_angle:  # turn positive
                vela = 17 - t_error + v_correction
                velb = -17 - t_error - v_correction
            else:  # turn negative
                vela = -17 - t_error - v_correction
                velb = 17 - t_error + v_correction

            #  target count calculation
            t_anglea += counta
            t_angleb += countb
            sleep(1)

            change_motors(vela, velb)

            right_turned = False
            left_turned = False
            lcounta = counta
            lcountb = countb
            t = 0
            while True:  # Turn
                if ((not back1 and counta >= t_anglea) or
                        (back1 and counta <= t_anglea)) and not right_turned:
                    right_turned = True
                    GPIO.output(inpa1, True)
                    GPIO.output(inpa2, True)
                    pwma.ChangeDutyCycle(0)
                if ((not back2 and countb >= t_angleb) or
                        (back2 and countb <= t_angleb)) and not left_turned:
                    left_turned = True
                    GPIO.output(inpb1, True)
                    GPIO.output(inpb2, True)
                    pwmb.ChangeDutyCycle(0)
                if right_turned and left_turned:
##                    print('angle target ' + str(t_phi*180/np.pi))
##                    print('angle est ' + str(theta_est*180/np.pi))
##                    print('target count a ' + str(t_anglea))
##                    print('count a ' + str(counta))
##                    print('target count b ' + str(t_angleb))
##                    print('count b ' + str(countb))
                    break
                sleep(0.001)
                if counta - lcounta == 0 and countb - lcountb == 0:
                    if t >= 1000:
                        print('low voltage correction')
                        v_correction += 3
                        change_motors(vela + v_correction, velb + v_correction)
                        t = 0
                    else:
                        t += 1
                else:
                    t = 0
                lcounta = counta
                lcountb = countb
            # stop the robot
            vela = velb = 0
        except ValueError:
            print('angle not expressed as a integer')
            vela = velb = 0

    else:  # Turn mode not activated
        if speed_angle == 1:
            strait = False
            vela = velb = 10
        elif speed_angle == 2:
            vela = 13 - output - error + v_correction
            velb = 13 + output + error + v_correction
            if vela < 0:
                vela = 0
            if velb < 0:
                velb = 0
            strait = True
        elif speed_angle == 3:
            vela = 13 - error + v_correction
            velb = 13 + error + v_correction
            for i in range(15):
                change_motors(i*vela/15, i*velb/15)
                sleep(0.1)
            strait = True
        else:
            strait = False
            vela = velb = 0
    change_motors(vela, velb)


def pose_control():  # corrects the pose to match the target one
    """
    :Input --> x_est: Current estimated x coordinate of the robot.
    :Input --> y_est: Current estimated y coordinate of the robot.
    :Input --> phi_est: Current estimated angle coordinate of the robot.
    :Output --> output: New PWM value for the right motor.
    This function also reads the global variables t_phi, t_x and t_y with the target pose of the robot.
    As the robot moves and turns the position gets altered by the angular speed when going strait or linear speed when
    turning. So the robot position deviates from the target one. This function deals with that. To avoid the error in
    the reads of the linear and angular speeds thy are not considered in the control, only the pose error
    """

    # Angle error
    phi_error = theta_est - t_phi
    if phi_error > np.pi:
        phi_error -= np.pi*2
    elif phi_error < -np.pi:
        phi_error += np.pi*2

    # pose error
    # point of the target line closer to the actual pose
    """
    This point is the intersection between the perpendicular line to the target line through the robot's pose and the
    target line.
    """
    if -0.001 < a < 0.001:
        if t_phi == np.pi/2:
            pose_error = -x_est+t_x
        elif t_phi == -np.pi/2:
            pose_error = x_est-t_x
        elif t_phi == 0:
            pose_error = y_est-t_y
        elif abs(t_phi) == np.pi:
            pose_error = -y_est+t_y
        else:
            pose_error = 0
    else:
        d = a*y_est - x_est
        print(a)
        target_x = (d-b)/(a-(1/a))
        target_y = a*target_x+b
        # robot pose
        t_pose = np.matrix([[x_est], [y_est], [0], [1]])
        # transform robot global coordinates to target point relative coordinates
        tro = np.matrix([[np.cos(t_phi), np.sin(t_phi), 0, -target_x*np.cos(t_phi)-target_y*np.sin(t_phi)],
                         [-np.sin(t_phi), np.cos(t_phi), 0, target_x*np.cos(t_phi)-target_y*np.sin(t_phi)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        t_pose = tro*t_pose
        pose_error = t_pose.item(1)

    # Inference Fuzzy controller

    # Fuzzify
    # angle error
    if phi_error >= np.pi/6:
        fuz_phi = np.matrix([0, 0, 0, 0, 1])
    elif phi_error >= np.pi/8:
        fuz_phi = np.matrix([0, 0, 0, (-8/np.pi)*(phi_error-(np.pi/6)), (8/np.pi)*(phi_error-(np.pi/8))])
    elif phi_error >= 0:
        fuz_phi = np.matrix([0, 0, 1-(phi_error/(np.pi/8)), (phi_error/(np.pi/8)), 0])
    elif phi_error >= -np.pi/8:
        fuz_phi = np.matrix([0, -(phi_error/(np.pi/8)), 1 + (phi_error/(np.pi/8)), 0, 0])
    elif phi_error >= np.pi/6:
        fuz_phi = np.matrix([0, 0, 0, (8/np.pi)*(phi_error+(np.pi/6)), (-8/np.pi)*(phi_error+(np.pi/8))])
    else:
        fuz_phi = np.matrix([1, 0, 0, 0, 0])
    # pose error
    if pose_error >= 10:
        fuz_pos = np.matrix([0, 0, 1])
    elif pose_error >= 0:
        fuz_pos = np.matrix([0, 1-(pose_error/10), (pose_error/10)])
    elif pose_error >= -10:
        fuz_pos = np.matrix([-(pose_error/10), 1+(pose_error/10), 0])
    else:
        fuz_pos = np.matrix([1, 0, 0])

    # Matching (product)
    matching = np.transpose(fuz_phi)*fuz_pos

    # implication (minimum)
    # doesn't need any calculation
    na = matching.item(0)
    pa = matching.item(14)

    # Defuzzification (weighted fuzzy mean)
    if na == 1:
        cna = -15 - v_correction 
    else:
        cna = (-20 - (-5-5*na))/2
    if pa == 1:
        cpa = 15 + v_correction
    else:
        cpa = (20 + (5*na+5))/2
    output = (cna*na + cpa*pa - 10*(matching.item(1) + matching.item(2) + matching.item(3) + matching.item(4) +
                                   matching.item(6)) +
              10*(matching.item(8) + matching.item(9) + matching.item(10) + matching.item(11) +
                  matching.item(13)))/(np.sum(matching))
    # output = int(round(phi_error*a1 + pose_error*a2 + a3*vang))
##    print('angle error ' + str(phi_error))
##    print('angle: ' + str(theta_est*180/np.pi))
##    print('angle target: ' + str(t_phi*180/np.pi))
##    print('pose error ' + str(pose_error))
##    print('ty: ' + str(t_y))
##    print('pos: ' + str(x_est) + str(y_est))
##    print('tx: ' + str(t_x))
##    print('a ' + str(a))
##    print(output)
    return output


def vision_jacobian(xr, yr, phir, gpose):
    h = np.matrix([[(matrix.item(0)*(yr - gpose.item(1) + ym*np.cos(phir) + xm*np.sin(phir))) /
                   (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                    gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2,
                   (matrix.item(0)*(gpose.item(0) - xr - xm*np.cos(phir) + ym*np.sin(phir))) /
                   (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                    gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2,
                   - (matrix.item(0)*(gpose.item(0)*np.cos(phir) - xr*np.cos(phir) +
                                      gpose.item(1)*np.sin(phir) - yr*np.sin(phir))) /
                   (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                    gpose.item(1)*np.sin(phir) + yr*np.sin(phir)) -
                   (matrix.item(0)*(gpose.item(1)*np.cos(phir) - yr*np.cos(phir) -
                                    gpose.item(0)*np.sin(phir) + xr*np.sin(phir)) *
                    (ym - gpose.item(1)*np.cos(phir) + yr*np.cos(phir) +
                     gpose.item(0)*np.sin(phir) - xr*np.sin(phir))) /
                   (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                    gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2],

                   [-(matrix.item(4)*np.cos(phir)*(gpose.item(2) - zm)) /
                    (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                     gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2,
                    -(matrix.item(4)*np.sin(phir)*(gpose.item(2) - zm)) /
                    (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                     gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2,
                    (matrix.item(4)*(gpose.item(2) - zm) *
                     (gpose.item(1)*np.cos(phir) - yr*np.cos(phir) -
                      gpose.item(0)*np.sin(phir) + xr*np.sin(phir))) /
                    (xm - gpose.item(0)*np.cos(phir) + xr*np.cos(phir) -
                     gpose.item(1)*np.sin(phir) + yr*np.sin(phir))**2]])
    return h


def robotcontrol(threadname):  # controls the robot and estimates its pose based on odometry.
    """
    :Input --> w1: Current angular speed of the right motor.
    :Input --> w2: Current angular speed of the left motor.
    :Input --> ww1: Target angular speed for the right motor.
    :Input --> ww2: Target angular speed for the left motor.
    :Input --> vela: current PWM for the right motor.
    :Input --> velb: current PWM for the left motor.
    :Output --> vela: New PWM value for the right motor.
    :Output --> velb: New PWM value fot the left motor
    This function controls the motors so that their target angular speed is reached. As the voltage for the motors is
    unknown the correct PWM is obtained by it's error between the target speed linear and angular speeds and the
    encoders readings. This function also makes the odometry based estimations of the robot pose.
    """
    file = open('data1.txt', 'w')
    file.write('velocidadlineal    Velocidadangular    nact\
                xest    yest    thetaest    control    xact    yact   theatact\
                    covariance' + '\n')

    # Test one:
    global actualization, x_act, y_act, theta_act, control, x_est, y_est, theta_est,\
        vlin, vang, nact, error, strait, done, v_correction

    ltime = 0  # Last time the encoders were read

    # Initial speeds
    vang = 0
    vlin = 0
    lcounta = 0
    lcountb = 0

    # Other
    count = 0  # times the linear speed is lower than the minimum
    out = 0

    done = False

    lnact = 0

    # kalman
    pk = ppk

    while control:  # Start control

        """
        # Read encoders
        Only update the time only if the count isn´t to avoid missing very low speeds ((((not done)))), speeds can't be
        that low. Calculate the angular speed of the motors and the linear and angular speed of the robot. There is only
        one encoder, so we can only read the speed not the direction. to correct this we adjust the direction to match
        the control set to the motors.

        doing this one count can be lost nad counted in the next step, as there are many marks 24 per revolution
        this isn`t a big issue
        """
        # Angular speed of the motors

        time = cv2.getTickCount()/cv2.getTickFrequency()
        da = counta - lcounta
        db = countb - lcountb
        # print('da: '+ str(da))
        # print('db: '+ str(db))
        # print('Count a: ' + str(counta))
        # print('Count b: ' + str(countb))

        lcounta = counta
        lcountb = countb

        w1 = (da*np.pi)/(marks*(time-ltime))
        w2 = (db*np.pi)/(marks*(time-ltime))

        """
        # get position actualization
        If the robots corrects its pose estimation using the camera sensor it has to be updated. This step is important
        as the odometry based pose estimation error grows as the robot moves. The pose actualization is applied to the
        last known pose. take care not to make the actualization after the new speed if calculated---------------------------------
        """

        # linear and angular speeds of the robot (not the target ones)
        # lvlin = vlin
        # lvang = vang
        vlin = r*(w1 + w2)/2
        vang = r*(w1 - w2)/l

        # Prediction phase:
        # pose prediction
        # Estimate the new robot pose

        y_est += (time-ltime) * vlin*np.sin(theta_est)
        x_est += (time-ltime) * vlin*np.cos(theta_est)

        y_act += (time-ltime) * vlin*np.sin(theta_act)
        x_act += (time-ltime) * vlin*np.cos(theta_act)

        theta_est += (time-ltime)*vang
        theta_act += (time-ltime)*vang

        if theta_est > np.pi:  # normalise theta_est
            theta_est -= 2*np.pi
        elif theta_est < -np.pi:
            theta_est += 2*np.pi
        if theta_act > np.pi:  # normalise theta_est
            theta_act -= 2*np.pi
        elif theta_act < -np.pi:
            theta_act += 2*np.pi

        pos = np.matrix([[x_est], [y_est], [theta_est]])
        # pose estimation covariance
        fk1 = np.matrix([[1, 0, -vlin*(time-ltime)*np.sin(theta_est)],
                         [0, 1, vlin*(time-ltime)*np.cos(theta_est)],
                         [0, 0, 1]])  # Jacobian
        pk1 = fk1*pk*np.transpose(fk1) + q  # covariance matrix
    
        # Actualization phase
        if nact - lnact >= 10:  # only check every 10 iterations
            lnact = nact
            # check frame
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            if ret:
                # correct frame
                h,  w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, matrix, distortion, None, newcameramtx)
                x,y,w,h = roi
                frame = frame[y:y+h, x:x+w]
                ret, length, image_center, real_center, fframe = localization.pattern(frame)
                if ret and length > 0:
                    print('LANDMARK detected¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡')
                    # estimate the image points given the global coordinates of the points
                    # and also its jacobian
                    tg_c = np.matrix([[np.sin(theta_est), -np.cos(theta_est), 0,
                                       ym + y_est*np.cos(theta_est) - x_est*np.sin(theta_est)],
                                      [0, 0, -1, zm],
                                      [np.cos(theta_est), np.sin(theta_est), 0,
                                       -xm - x_est*np.cos(theta_est) - y_est*np.sin(theta_est)],
                                      [0, 0, 0, 1]])
                    # print(real_center)
                    dd = False
                    for i in real_center:
                        # get camera coordinates
                        c = tg_c*np.transpose(i)
                        # print(c)
                        # project to image
                        u = matrix.item(2) + matrix.item(0)*(c.item(0)/c.item(2))
                        v = matrix.item(5) + matrix.item(4)*(c.item(1)/c.item(2))
                        if dd:
                            est_im_centers = np.vstack([est_im_centers, [u], [v]])
                            hk = np.vstack([hk, vision_jacobian(x_est, y_est, theta_est, i)])
                        else:
                            est_im_centers = np.matrix([[u], [v]])
                            hk = vision_jacobian(x_est, y_est, theta_est, i)
                            dd = True

                    # innovation
                    print('img: ' + str(image_center))
                    yk = image_center - est_im_centers
                    # Innovation covariance
                    length, wide = np.shape(yk)
                    print('yk: ' + str((yk)))
                    # yk = yk.reshape((2*length,1))
                    rk = np.identity(length)*var_v
                    sk = hk*pk1*np.transpose(hk) + rk
                    # Kalman gain
                    kk = pk1*np.transpose(hk)*np.linalg.inv(sk)
                    print(kk)
                    # Correct the estimation
                    print('pos antes' + str(pos))
                    pos = pos + kk*(yk)
                    print('pos despues' + str(pos))
                    print('pos sin' + str([x_act, y_act, theta_act]))
                    x_est = pos.item(0)
                    y_est = pos.item(1)
                    theta_est = pos.item(2)
                    # Covariance actualization
                    pk = (np.identity(3)-kk*hk)*pk1
                else:
                    pk = pk1
            else:
                pk = pk1

        # error correction
        if strait:
            # voltage correction
            if vlin < 10:  # if the linear speed is lower than 10 cm/s the voltage is low
                print('LOW VOLTAGE' + str(count))
                if count >= 4:
                    v_correction += 3
                    count = 0
                else:
                    count += 1
            else:
                count = 0
            if vlin > 24:  # if the linear speed is too high the speed must be adjusted
                v_correction -= 3
            # robot control
            output = pose_control()
            change_speed(False, 2, output, 0, 0)

        file.write(str(vlin)+'    '+str(vang) +
                   '    '+str(nact)+'    '+str(x_est) +
                   '    '+str(y_est)+'    '+str(theta_est) +
                   '    ' + str(out) +
                   '    ' + str(x_act) +
                   '    ' + str(y_act) +
                   '    ' + str(theta_act) +
                   '    ' + str(pk) + '\n')
        ltime = time
        nact += 1
        sleep(0.2)

    done = True
    file.close()


"""
Camera navigation functions
"""


def object_check(normal):  # This function uses the floorFilter() function to see if there are objects in the way
    if normal:  # normal use of the function stop the robot if something is detected
        ret, frame = cap.read()  # capture frame ret --> boolean to check if the frame is correct
        ret, frame = cap.read()  # throw some frames to take the last one
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        if ret:  # check if the frame is correctly read
            det, what = basicnavigation.floorFilter(frame)
            print('floorfilterwhat:' + str(what))
            return det, what
        else:
            return True, 'nno'
    else:  # not usual operation now the robot is still waiting to se if there are any objects
        while True:
            ret, frame = cap.read()  # capture frame ret --> boolean to check if the frame is correct
            ret, frame = cap.read()  # throw some frames to take the last one
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            if ret:
                det, what = basicnavigation.floorFilter(frame)  # check again
            else:
                what = 'nno'
            ret, frame = cap.read()  # capture frame ret --> boolean to check if the frame is correct
            ret, frame = cap.read()  # throw some frames to take the last one
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            if ret:
                det, whatt = basicnavigation.floorFilter(frame)  # check again
            else:
                whatt = 'nno'
                det = True
            if what is whatt:
                return det, what


def obstacle_avoidance():  # this function deals with avoiding objects in the way
    det, what = object_check(True)
    print(what)
    if det:
        t_pose = target_position()
        change_speed(False, 0, 0, t_pose.item(0), t_pose.item(1))
        sleep(0.5)
        change_speed(False, 3, 0, t_pose.item(0), t_pose.item(1))
        while True:
            if np.sqrt((x_est-t_pose.item(0))**2 + (y_est - t_pose.item(1))**2) >= 10:
                break
            sleep(0.001)
        while True:
            print('loooooooooooooooooooooooooooooooooooooooooooooooooooooooooop')
            if what == 'nno':  # camera not detected
                change_speed(False, 0, 0, 0, 0)
                break
            elif (what == 'left') or (what == 'middle') or (what == 'both'):
                print('dentro1')
                sleep(0.5)
                # turn right
                t_pose = target_position()
                change_speed(True, -np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                det, what = object_check(False)
                if det:  # something on the way
                    sleep(0.5)
                    change_speed(True, np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                    what = 'right'
                    print('right2')
                    continue
                else:
                    print('avanza1')
                    n = 1
                    while True:
                        change_speed(False, 3, 0, 0, 0)
                        while True:
                            if np.sqrt((x_est-t_pose.item(0))**2 + (y_est - t_pose.item(1))**2) >= 40*n:
                                    break
                            sleep(0.001)
                        sleep(0.5)
                        t_pose = target_position()
                        change_speed(True, np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                        det, what = object_check(False)
                        if det:
                            sleep(0.5)
                            change_speed(True, -np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                            n += 1
                            if n > 3:
                                break
                            continue
                        else:
                            break
                    change_speed(False, 3, 0, 0, 0)
                    break
            elif what == 'right':
                print('dentro2')
                sleep(0.5)
                # turn left
                t_pose = target_position()
                change_speed(True, np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                det, what = object_check(False)
                if det:  # something on the way
                    sleep(0.5)
                    change_speed(True, -np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                    what = 'left'
                    print('left2')
                    continue
                else:
                    print('avanza2')
                    n = 1
                    while True:
                        change_speed(False, 3, 0, 0, 0)
                        while True:
                            if np.sqrt((x_est-t_pose.item(0))**2 + (y_est - t_pose.item(1))**2) >= 25*n:
                                    break
                            sleep(0.001)
                        sleep(0.5)
                        t_pose = target_position()
                        change_speed(True, -np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                        det, what = object_check(False)
                        if det:
                            sleep(0.5)
                            change_speed(True, np.pi/2, 0, t_pose.item(0), t_pose.item(1))
                            n += 1
                            if n > 3:
                                break
                            continue
                        else:
                            break
                    change_speed(False, 3, 0, 0, 0)
                    break


def video_capture(channel):
    while control:
        rert, frame = cap.read()
        if rert:
            out.write(frame)
            # ret, what = basicnavigation.floorFilter(frame)
            ret, length, image_center, real_center, fframe = localization.pattern(frame)
            out2.write(fframe)
        sleep(0.0001)

# MAIN

# Set up the GPIO Control function

# Only test mode
nact = 0

# Initial speeds
speed_angle = 0
turn = False
change = True

# Odometry correction
actualization = False
x_act = 0
y_act = 0
theta_act = 0

# Control parameters
control = True
done = False

# start the thread
start_new_thread(robotcontrol, ('Robot control',))
##start_new_thread(video_capture, ('video capture',))

# File set up for test mode

try:
    sleep(5)
    k = 3
    while k <= 5:
        # go strait
        change_speed(False, 3, 0, 0, 0)
        while x_est <= 100:
            # print('hola desde fuera')
            # obstacle_avoidance()
            # k +=1
            sleep(0.1)
        change_speed(False, 0, 0, 0, 0)
        f = input('>')
        # sleep(0.5)
        # turn
        change_speed(True, -np.pi/2, 0, 100, 0)
        sleep(0.5)
        # go strait
        change_speed(False, 3, 0, 0, 0)
        while y_est >= -50:
            sleep(0.01)
        change_speed(False, 0, 0, 0, 0)
        f = input('>')
        # sleep(0.5)
        # turn
        change_speed(True, -np.pi/2, 0, 100, -50)
        sleep(0.5)
        # go strait
        change_speed(False, 3, 0, 0, 0)
        while x_est >= 0:
            sleep(0.01)
        change_speed(False, 0, 0, 0, 0)
        f = input('>')
        # sleep(0.5)
        # turn
        change_speed(True, -np.pi/2, 0, 0, -50)
        sleep(0.5)
        # go strait
        change_speed(False, 3, 0, 0, 0)
        while y_est <= 0:
            sleep(0.01)
        change_speed(False, 0, 0, 0, 0)
        f = input('>')
        # sleep(0.5)
        # turn
        change_speed(True, -np.pi/2, 0, 0, 0)
        k += 1
    sleep(2)
except KeyboardInterrupt:
    pass

# End the GPIO control
control = False
end_gpio()
# End camera
cv2.destroyAllWindows()
cap.release()
out.release()
out2.release()

sleep(4)
print('error: ' + str(error))
print('t_error: ' + str(t_error))
print('voltage: ' + str(v_correction))
print('angle: ' + str(theta_est * 180/np.pi))
print('pos con' + str([x_est, y_est, theta_est]))
print('pos sin' + str([x_act, y_act, theta_act]))
print('end')
