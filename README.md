#The Open Motion Planning Library (OMPL)
=======================================

##OS
Linux: Ubuntu 14.04
ROS: Indigo

## Dependency
OMPL has the following required dependencies:

* [Boost](http://www.boost.org) (version 1.48 or higher)
* [CMake](http://www.cmake.org) (version 2.8.7 or higher)

The following dependencies are optional:

* [ODE](http://ode.org) (needed to compile support for planning using the Open Dynamics Engine)
* [Py++](https://bitbucket.org/ompl/pyplusplus) (needed to generate Python bindings)
* [Doxygen](http://www.doxygen.org) (needed to create a local copy of the documentation at
  http://ompl.kavrakilab.org/core)
* [Eigen](http://eigen.tuxfamily.org) (needed for an informed sampling technique to improve the optimization of path length)

## Compile and build
Once dependencies are installed, you can build OMPL on Linux, OS X,
and MS Windows. Go to the top-level directory of OMPL and type the
following commands:

    mkdir -p build/Release
    cd build/Release
    cmake ../..
    # next two steps are optional
    make installpyplusplus && cmake . # download & install Py++
    make update_bindings # if you want Python bindings
    make -j 4 # replace "4" with the number of cores on your machine

The above is the installation guide from official repository but since we modified and added new planner and state space to ompl, the installation process will be a little bit tricky because MoveIt! and ompl depend on each other right now.
```
1. Follow the official tutorial to install source version of MoveIt!, binary verison of ompl will be installed during the installation.
2. go to your local folder to remove the installed binary ompl.
3. Install source modified ompl as follows:
   cd path_to_ompl
   mkdir -p build/Release
   cd build/Release
   cmake DCMAKE_INSTALL_PREFIX=/opt/ros/indigo ../..
   make -j4
   sudo make install
4. remove the official MoveIt! package, git clone ivaMoveitCore repository and do catkin_make
```
