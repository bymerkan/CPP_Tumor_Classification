# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/computer/Documents/torch-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/computer/Documents/torch-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/torch-cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torch-cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torch-cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torch-cpp.dir/flags.make

CMakeFiles/torch-cpp.dir/src/main.cpp.o: CMakeFiles/torch-cpp.dir/flags.make
CMakeFiles/torch-cpp.dir/src/main.cpp.o: /Users/computer/Documents/torch-cpp/src/main.cpp
CMakeFiles/torch-cpp.dir/src/main.cpp.o: CMakeFiles/torch-cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/computer/Documents/torch-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torch-cpp.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch-cpp.dir/src/main.cpp.o -MF CMakeFiles/torch-cpp.dir/src/main.cpp.o.d -o CMakeFiles/torch-cpp.dir/src/main.cpp.o -c /Users/computer/Documents/torch-cpp/src/main.cpp

CMakeFiles/torch-cpp.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/torch-cpp.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/computer/Documents/torch-cpp/src/main.cpp > CMakeFiles/torch-cpp.dir/src/main.cpp.i

CMakeFiles/torch-cpp.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/torch-cpp.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/computer/Documents/torch-cpp/src/main.cpp -o CMakeFiles/torch-cpp.dir/src/main.cpp.s

CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o: CMakeFiles/torch-cpp.dir/flags.make
CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o: /Users/computer/Documents/torch-cpp/src/customDataset.cpp
CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o: CMakeFiles/torch-cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/computer/Documents/torch-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o -MF CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o.d -o CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o -c /Users/computer/Documents/torch-cpp/src/customDataset.cpp

CMakeFiles/torch-cpp.dir/src/customDataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/torch-cpp.dir/src/customDataset.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/computer/Documents/torch-cpp/src/customDataset.cpp > CMakeFiles/torch-cpp.dir/src/customDataset.cpp.i

CMakeFiles/torch-cpp.dir/src/customDataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/torch-cpp.dir/src/customDataset.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/computer/Documents/torch-cpp/src/customDataset.cpp -o CMakeFiles/torch-cpp.dir/src/customDataset.cpp.s

# Object files for target torch-cpp
torch__cpp_OBJECTS = \
"CMakeFiles/torch-cpp.dir/src/main.cpp.o" \
"CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o"

# External object files for target torch-cpp
torch__cpp_EXTERNAL_OBJECTS =

torch-cpp: CMakeFiles/torch-cpp.dir/src/main.cpp.o
torch-cpp: CMakeFiles/torch-cpp.dir/src/customDataset.cpp.o
torch-cpp: CMakeFiles/torch-cpp.dir/build.make
torch-cpp: /usr/local/lib/libopencv_gapi.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_stitching.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_alphamat.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_aruco.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_bgsegm.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_bioinspired.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_ccalib.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_dnn_objdetect.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_dnn_superres.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_dpm.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_face.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_freetype.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_fuzzy.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_hfs.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_img_hash.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_intensity_transform.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_line_descriptor.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_mcc.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_quality.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_rapid.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_reg.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_rgbd.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_saliency.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_sfm.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_stereo.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_structured_light.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_superres.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_surface_matching.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_tracking.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_videostab.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_viz.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_wechat_qrcode.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_xfeatures2d.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_xobjdetect.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_xphoto.4.8.0.dylib
torch-cpp: /Users/computer/Documents/torch-cpp/libtorch/lib/libc10.dylib
torch-cpp: /Users/computer/Documents/torch-cpp/libtorch/lib/libkineto.a
torch-cpp: /usr/local/lib/libopencv_shape.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_highgui.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_datasets.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_plot.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_text.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_ml.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_phase_unwrapping.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_optflow.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_ximgproc.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_video.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_videoio.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_imgcodecs.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_objdetect.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_calib3d.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_dnn.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_features2d.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_flann.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_photo.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_imgproc.4.8.0.dylib
torch-cpp: /usr/local/lib/libopencv_core.4.8.0.dylib
torch-cpp: /Users/computer/Documents/torch-cpp/libtorch/lib/libtorch.dylib
torch-cpp: /Users/computer/Documents/torch-cpp/libtorch/lib/libtorch_cpu.dylib
torch-cpp: /Users/computer/Documents/torch-cpp/libtorch/lib/libc10.dylib
torch-cpp: CMakeFiles/torch-cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/computer/Documents/torch-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable torch-cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torch-cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torch-cpp.dir/build: torch-cpp
.PHONY : CMakeFiles/torch-cpp.dir/build

CMakeFiles/torch-cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torch-cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torch-cpp.dir/clean

CMakeFiles/torch-cpp.dir/depend:
	cd /Users/computer/Documents/torch-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/computer/Documents/torch-cpp /Users/computer/Documents/torch-cpp /Users/computer/Documents/torch-cpp/build /Users/computer/Documents/torch-cpp/build /Users/computer/Documents/torch-cpp/build/CMakeFiles/torch-cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/torch-cpp.dir/depend

