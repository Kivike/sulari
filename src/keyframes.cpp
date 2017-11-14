#include "keyframes.h"

KeyFrames::KeyFrames(int keyFrameCount, int* keyFrames, bool humanInFrame) {
    this->keyFrameCount = keyFrameCount;
    this->keyFrames = keyFrames;
    this->humanInFrame = humanInFrame;
}

bool KeyFrames::isHumanInFrame(int frame) {
    for(int i = 0; i < this->keyFrameCount; i++) {
        if(this->keyFrames[i] == frame) {
            this->humanInFrame = !this->humanInFrame;
            break;
        }
    }
    return this->humanInFrame;
}
