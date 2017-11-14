#ifndef KEYFRAMES_H
#define KEYFRAMES_H

class KeyFrames {
public:
    KeyFrames(int, int*, bool);
    ~KeyFrames() {
        delete keyFrames;
    }
    bool isHumanInFrame(int);
private:
    int keyFrameCount;
    int *keyFrames;
    bool humanInFrame;
};

#endif
