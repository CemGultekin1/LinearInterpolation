#!/bin/bash

# Specify the source file and output executable name
SOURCE_FILE="main.cpp"
OUTPUT_EXECUTABLE="main"

# Compile the source file with C++20 support
clang++ -std=c++20 -o "$OUTPUT_EXECUTABLE" "$SOURCE_FILE"

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable created: $OUTPUT_EXECUTABLE"
else
    echo "Compilation failed."
fi
