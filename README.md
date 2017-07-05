# sulari
Sulautettujen järjestelmien projekti

### Running tests
> make
> ./sulari -test

### Algorithm parameters:
**LBP::PIXEL_VALUE_TOLERANCE** How much can pixel grayscale value differ while still being considered the same
**LBP::HISTOGRAM_REGION_SIZE** How big area does histogram describe
**LBP::NEIGHBOUR_COUNT** How many neighbour pixels are used to calculate Local Binary Pattern for a pixel
**LBP::DESCRIPTOR_RADIUS** How far away are neighbours pixels (uses interpolation)
**LBPPixel::BACKGROUND_WEIGHT** How much weights does pixel need to have to be considered background (0.0 - 1.0)
**LBPPixel::HISTOGRAM_PROXIMITY_THRESHOLD** How close must histograms be to be considered close to each other (0.0 - 1.0)
**BackgroundRemover::BOUNDING_BOX_PADDING** How much padding is added to foreground box to give classifier more detection space
