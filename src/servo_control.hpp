#include <pigpio.h>

namespace {
    const unsigned int PAN_PIN = 13;
    const unsigned int TILT_PIN = 12;

    const unsigned int LOWER_BOUND_HW = 500;
    const unsigned int UPPER_BOUND_HW = 2500;
 
    const unsigned int LOWER_BOUND_PAN_HW = LOWER_BOUND_HW;
    const unsigned int UPPER_BOUND_PAN_HW = UPPER_BOUND_HW;

    const unsigned int LOWER_BOUND_TILT_HW = 700;
    const unsigned int UPPER_BOUND_TILT_HW = 1800;

    const int MAX_STEP_HW = 30;
    
    const float LOWER_BOUND_ANGLE = 0;
    const float UPPER_BOUND_ANGLE = 180;
    
    const float IDLE_TIME = 0.05f;
    const float IDLE_TIME_LONG = 0.5f;
    
    template<typename T>
    bool inRange(T val, T rangeMin, T rangeMax) {
        return val >= rangeMin && val <= rangeMax;
    }
    
    unsigned int toHWTicks(float angle) {
        unsigned int ticksInAngle = (UPPER_BOUND_HW - LOWER_BOUND_HW) / (UPPER_BOUND_ANGLE - LOWER_BOUND_ANGLE);
        return (unsigned int) (500 + angle * ticksInAngle);
    }
}


class PanTiltController {
public:
    bool init();
    void cleanup();
    
    void setPanAngle(float angle);
    void setTiltAngle(float angle);

    void handlePanShift(float shiftPercent);
    void handleTiltShift(float shiftPercent);

    void moveRight();
    void moveLeft();

    void moveUp();
    void moveDown();

private:
    unsigned int _currentPanHW = (UPPER_BOUND_PAN_HW - LOWER_BOUND_PAN_HW) / 2;
    unsigned int _currentTiltHW = (UPPER_BOUND_TILT_HW - LOWER_BOUND_TILT_HW) / 2;
};
