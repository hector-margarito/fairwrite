#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
/* This driver reads raw data from the BNO055
Connections
===========
Connect SCL to analog 5
Connect SDA to analog 4
Connect VDD to 3.3V DC
Connect GROUND to common ground
History
=======
2015/MAR/03 - First release (KTOWN)
*/
/* Set the delay between fresh samples */
#define BNO055_SAMPLERATE_DELAY_MS (15)
Adafruit_BNO055 bno = Adafruit_BNO055();
unsigned long time_stamp;
int inPin = 9; // switch connected to digital pin 9
int val = 0; // variable to store the read value
/**************************************************************************/
/*
Arduino setup function (automatically called at startup)
*/
/**************************************************************************/
void setup(void)
{
    Serial.begin(115200);
    Serial.println("Orientation Sensor Raw Data Test"); Serial.println("");
    /* Initialise the sensor */
    if(!bno.begin())
    {
        /* There was a problem detecting the BNO055 ... check your connections */
        Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        while(1);
    }
    delay(1000);
    pinMode(inPin, INPUT); // sets the digital pin 9 as input
    /* Display the current temperature */
    int8_t temp = bno.getTemp();
    Serial.print("Current Temperature: ");
    Serial.print(temp);
    Serial.println(" C");
    Serial.println("");
    bno.setExtCrystalUse(true);
    Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");
}
/**************************************************************************/
/*
Arduino loop function, called once 'setup' is complete (your own code
should go here)
*/
/**************************************************************************/
void loop(void)
{   
    // Possible vector values can be:
    // - VECTOR_ACCELEROMETER - m/s^2
    // - VECTOR_MAGNETOMETER - uT
    // - VECTOR_GYROSCOPE - rad/s
    // - VECTOR_EULER - degrees
    // - VECTOR_LINEARACCEL - m/s^2
    // - VECTOR_GRAVITY - m/s^2
    
    /*** First Obtain all information before printing to avoid mismatch between
         values obtained at different time ***/
    time_stamp = millis();
    uint8_t sys, gyro, accel, mag = 0;
    imu::Vector<3> accelerometer = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gyroscope     = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Vector<3> magnetometer  = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
    imu::Quaternion quat         = bno.getQuat();
    bno.getCalibration(&sys, &gyro, &accel, &mag);
    val = digitalRead(inPin); // read the input pin
    
    /* Display time stamp */
    Serial.print(time_stamp);
    Serial.print(" ");
    /* Display switch state */
    Serial.print(val);
    Serial.print(" ");
    /* Display calibration state */
    Serial.print(accel);
    Serial.print(gyro);
    Serial.print(mag);
    Serial.print(sys);
    Serial.print(" ");
    /* Display the floating point data */
    Serial.print("ACC: ");
    Serial.print(accelerometer.x());
    Serial.print(" ");
    Serial.print(accelerometer.y());
    Serial.print(" ");
    Serial.print(accelerometer.z());
    /* Display the floating point data */
    Serial.print(" GYR: ");
    Serial.print(gyroscope.x());
    Serial.print(" ");
    Serial.print(gyroscope.y());
    Serial.print(" ");
    Serial.print(gyroscope.z());
    /* Display the floating point data */
    Serial.print(" MAG: ");
    Serial.print(magnetometer.x());
    Serial.print(" ");
    Serial.print(magnetometer.y());
    Serial.print(" ");
    Serial.print(magnetometer.z());
    /* Quaternion data */
    Serial.print(" QUAT: ");
    Serial.print(quat.w(), 4);
    Serial.print(" ");
    Serial.print(quat.x(), 4);
    Serial.print(" ");
    Serial.print(quat.y(), 4);
    Serial.print(" ");
    Serial.println(quat.z(), 4);
    
    delay(BNO055_SAMPLERATE_DELAY_MS);
}
