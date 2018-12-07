# Auto-KeyStone-Correction

This program corrects the keystone distortion of a projector automatically. \
It detects the surface that projected onto and find its equation. \
Plane detection is done by using OpenARK library (https://github.com/augcog/OpenARK) \

To run this program \
Build it by using given CMakelist.\
It requires OpenCV, Boost, and Royale library to connect PMD pico-flexx camera. \
After build it, connect your projector and camera. \
And run by command ./keystone \
You can specify the exposure times ./keystone -e 100 \
If exposure is not specified, auto-exposure mode is used \
In order that plane detection works, auto-exposore mode is suggested. \
