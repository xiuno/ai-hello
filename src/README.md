find /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -name cstdint

find /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk -name cstdint
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1/cstdint

export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
export CPATH=$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include
export CPLUS_INCLUDE_PATH=$SDKROOT/usr/include/c++/v1:$SDKROOT/usr/include
