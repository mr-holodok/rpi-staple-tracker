#ifndef CAPTURESETTINGS_H
#define CAPTURESETTINGS_H

struct CaptureSettings
{
public:
    CaptureSettings(int w, int h, int fps) :
        _width{w}, _height{h}, _FPS{fps} {}

    int width()
    {
        return _width;
    }

    int height() {
        return _height;
    }

    int FPS() {
        return _FPS;
    }

private:
    int _width, _height, _FPS;
};

#endif // CAPTURESETTINGS_H
