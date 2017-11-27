#ifndef CONFIG_H
#define CONFIG_H

class Config
{
public:
    static const bool COMBINE_FRAMES = true;
    static const bool BGR_INTERLACE_ENABLED = true;
    static const unsigned int BGR_INTERLACE_EVERY = 4;
    static const int BGR_BOUNDING_BOX_PADDING = 12;

    // How close to each other can pixel gray-scale values be
    // while still considering them the same
    static const int LBP_PIXEL_VALUE_TOLERANCE = 10;
    // Currently the region is a X*X square
    static const int LBP_HISTOGRAM_REGION_SIZE = 14;
    // If set to true, only every other half of rows are handled on each frame
    static const unsigned int LBP_NEIGHBOUR_COUNT = 6;
    static const unsigned int LBP_DESCRIPTOR_RADIUS = 3;

    // 0.0f - 1.0f
    static constexpr float PIXEL_BACKGROUND_WEIGHT = 0.5f;

    // How close must histograms be to each other for them to be considered
    // similar
    static constexpr float HISTOGRAM_PROXIMITY_THRESHOLD = 0.90f; // 0.9f
};
#endif
