# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/giannis/ClionProjects/NanoFlannKDTree

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giannis/ClionProjects/NanoFlannKDTree/build

# Include any dependencies generated for this target.
include CMakeFiles/NanoFlannKDTree.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NanoFlannKDTree.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NanoFlannKDTree.dir/flags.make

CMakeFiles/NanoFlannKDTree.dir/main.cpp.o: CMakeFiles/NanoFlannKDTree.dir/flags.make
CMakeFiles/NanoFlannKDTree.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giannis/ClionProjects/NanoFlannKDTree/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NanoFlannKDTree.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NanoFlannKDTree.dir/main.cpp.o -c /home/giannis/ClionProjects/NanoFlannKDTree/main.cpp

CMakeFiles/NanoFlannKDTree.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NanoFlannKDTree.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giannis/ClionProjects/NanoFlannKDTree/main.cpp > CMakeFiles/NanoFlannKDTree.dir/main.cpp.i

CMakeFiles/NanoFlannKDTree.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NanoFlannKDTree.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giannis/ClionProjects/NanoFlannKDTree/main.cpp -o CMakeFiles/NanoFlannKDTree.dir/main.cpp.s

CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.requires

CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.provides: CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/NanoFlannKDTree.dir/build.make CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.provides

CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.provides.build: CMakeFiles/NanoFlannKDTree.dir/main.cpp.o


# Object files for target NanoFlannKDTree
NanoFlannKDTree_OBJECTS = \
"CMakeFiles/NanoFlannKDTree.dir/main.cpp.o"

# External object files for target NanoFlannKDTree
NanoFlannKDTree_EXTERNAL_OBJECTS =

NanoFlannKDTree: CMakeFiles/NanoFlannKDTree.dir/main.cpp.o
NanoFlannKDTree: CMakeFiles/NanoFlannKDTree.dir/build.make
NanoFlannKDTree: CMakeFiles/NanoFlannKDTree.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giannis/ClionProjects/NanoFlannKDTree/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable NanoFlannKDTree"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NanoFlannKDTree.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NanoFlannKDTree.dir/build: NanoFlannKDTree

.PHONY : CMakeFiles/NanoFlannKDTree.dir/build

CMakeFiles/NanoFlannKDTree.dir/requires: CMakeFiles/NanoFlannKDTree.dir/main.cpp.o.requires

.PHONY : CMakeFiles/NanoFlannKDTree.dir/requires

CMakeFiles/NanoFlannKDTree.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NanoFlannKDTree.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NanoFlannKDTree.dir/clean

CMakeFiles/NanoFlannKDTree.dir/depend:
	cd /home/giannis/ClionProjects/NanoFlannKDTree/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giannis/ClionProjects/NanoFlannKDTree /home/giannis/ClionProjects/NanoFlannKDTree /home/giannis/ClionProjects/NanoFlannKDTree/build /home/giannis/ClionProjects/NanoFlannKDTree/build /home/giannis/ClionProjects/NanoFlannKDTree/build/CMakeFiles/NanoFlannKDTree.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NanoFlannKDTree.dir/depend

