#include <cmath>
#include <iostream>
#include "servo_control.hpp"

bool PanTiltController::init() {
    return (gpioInitialise() >= 0);
}

void PanTiltController::cleanup() {
    gpioServo(TILT_PIN, 0);
    gpioServo(PAN_PIN, 0);
    gpioTerminate();
}

void PanTiltController::setPanAngle(float angle) {
    if (inRange(angle, LOWER_BOUND_ANGLE, UPPER_BOUND_ANGLE)) {
        unsigned int targetHW = toHWTicks(angle);
	    //unsigned int target = targetHW;
        //if (std::abs((int)targetHW - (int)_currentPanHW) > MAX_STEP_HW) {
        //    target = targetHW > _currentPanHW ? _currentPanHW + MAX_STEP_HW : _currentPanHW - MAX_STEP_HW;
        //}
	    //if (!inRange(targetHW, _currentPanHW - 25, _currentPanHW + 25)) {
        gpioServo(PAN_PIN, targetHW);
        _currentPanHW = targetHW;
	    //}
        time_sleep(IDLE_TIME_LONG);
    }
}

void PanTiltController::setTiltAngle(float angle) {
    if (inRange(angle, LOWER_BOUND_ANGLE, UPPER_BOUND_ANGLE)) {
        unsigned int targetHW = toHWTicks(angle);
        //if (std::abs((int)targetHW - (int)_currentTiltHW) > MAX_STEP_HW) {
        //    targetHW = targetHW > _currentTiltHW ? _currentTiltHW + MAX_STEP_HW : _currentTiltHW - MAX_STEP_HW;
        //}
        gpioServo(TILT_PIN, targetHW);
        _currentTiltHW = targetHW;
        time_sleep(IDLE_TIME_LONG);
    }
}

void PanTiltController::moveRight() {
    if (_currentPanHW - MAX_STEP_HW >= LOWER_BOUND_PAN_HW) {
    	_currentPanHW -= MAX_STEP_HW;
	gpioServo(PAN_PIN, _currentPanHW);
	time_sleep(IDLE_TIME);
	gpioServo(PAN_PIN, 0);
    }
}

void PanTiltController::moveLeft() {
    if (_currentPanHW + MAX_STEP_HW <= UPPER_BOUND_PAN_HW) {
    	_currentPanHW += MAX_STEP_HW;
	gpioServo(PAN_PIN, _currentPanHW);
	time_sleep(IDLE_TIME);
	gpioServo(PAN_PIN, 0);
    }
}

void PanTiltController::moveUp() {
    if (_currentTiltHW - MAX_STEP_HW >= LOWER_BOUND_TILT_HW) {
    	_currentTiltHW -= MAX_STEP_HW;
	gpioServo(TILT_PIN, _currentTiltHW);
	time_sleep(IDLE_TIME);
	gpioServo(TILT_PIN, 0);
    }
}

void PanTiltController::moveDown() {
    if (_currentTiltHW + MAX_STEP_HW <= UPPER_BOUND_TILT_HW) {
    	_currentTiltHW += MAX_STEP_HW;
	gpioServo(TILT_PIN, _currentTiltHW);
	time_sleep(IDLE_TIME);
	gpioServo(TILT_PIN, 0);
    }
}

// shiftPercent - delta between centers / frame width
void PanTiltController::handlePanShift(float shiftPercent) {
    if (std::abs(shiftPercent) <= 3.0f) {
    	time_sleep(IDLE_TIME);
	return;
    }
    int posShift = /*std::abs(shiftPercent) > 25.f ? 
        (std::signbit(shiftPercent) ? -25 * 1.2 : 25 * 1.2) :*/
        (int) shiftPercent * 1.0;
    if (!inRange(_currentPanHW + posShift, LOWER_BOUND_PAN_HW, UPPER_BOUND_PAN_HW))
        _currentPanHW = _currentPanHW <= LOWER_BOUND_PAN_HW ? LOWER_BOUND_PAN_HW : UPPER_BOUND_PAN_HW;
    else
        _currentPanHW += posShift;
    gpioServo(PAN_PIN, _currentPanHW);
    time_sleep(IDLE_TIME);
    gpioServo(PAN_PIN, 0);
}

void PanTiltController::handleTiltShift(float shiftPercent) {
    if (std::abs(shiftPercent) <= 5.0f) {
    	time_sleep(IDLE_TIME);
	return;
    }
    int posShift = /* std::abs(shiftPercent) > 10.f ? 
        (int) ((std::signbit(shiftPercent) ? -2 : 2) * 10) : */
        (int) (shiftPercent * 1);
    if (!inRange(_currentTiltHW - posShift, LOWER_BOUND_TILT_HW, UPPER_BOUND_TILT_HW))
        _currentTiltHW = _currentTiltHW - posShift <= LOWER_BOUND_TILT_HW ? LOWER_BOUND_TILT_HW : UPPER_BOUND_TILT_HW;
    else
        _currentTiltHW -= posShift;
    gpioServo(TILT_PIN, _currentTiltHW);
    time_sleep(IDLE_TIME);
    gpioServo(TILT_PIN, 0);
}
