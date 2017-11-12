/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Leo Keselman */

/*
	Here we are defining a class which allows us to get the jacobian and FK of some certain robot
	This is a very temporary solution only. It applies specifically to MoveIt only.
	A more permanent solution would be a base class that defines as virtual functions the necessary
	functions of the class. Then a class like this should inherit from it and redefine the
	functions properly
*/

#ifndef OMPL_BASE_SPACES_MANIPULATOR_STATE_
#define OMPL_BASE_SPACES_MANIPULATOR_STATE_

#include <ros/ros.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <Eigen/Core>
#include <Eigen/Geometry>


class ManipulatorState
{
	public:
		/* \brief Constructor for ManipulatorState. So far nothing happens */
		ManipulatorState(void)
		{

		}

		~ManipulatorState(void)
		{

		}

		/* \brief Initialize manipulator state */
		void Initialize(std::string robotDescription, std::string jointGroupName)
                {

                        /* Load the robot model */
                        robot_model_loader::RobotModelLoader robot_model_loader(robotDescription);

                        /* Get a shared pointer to the model */
                        kinematic_model = robot_model_loader.getModel();

                        /* Get and print the name of the coordinate frame in which the transforms for this model are computed*/
                        ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());

                        /* Create a kinematic state - this represents the configuration for the robot represented by kinematic_model */
                        kinematic_state = robot_state::RobotStatePtr(new robot_state::RobotState(kinematic_model));

                        /* Set all joints in this state to their default values */
                        kinematic_state->setToDefaultValues();

                        joint_model_group = kinematic_model->getJointModelGroup(jointGroupName);
		
		}

	        /* \brief Return end effector pose as 6x1 vector (x,y,z,Rx,Ry,Rz) */
		Eigen::VectorXd getEefPose(Eigen::VectorXd jointState)
		{	
        		//Set the kinematic state to have the proper joint values
        		kinematic_state->setJointGroupPositions(joint_model_group, jointState);

        		//Retrieve the end-effector state as an affine (homogeneous) transform
        		const Eigen::Affine3d &end_effector_state = kinematic_state->
        		getGlobalLinkTransform(kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back()));

        		//If the rotation should be a quaternion vector, the line below should be uncommented
        		Eigen::Quaterniond quaternion(end_effector_state.rotation());

        		//Otherwise, we use Euler angles for rotation
        		//Eigen::VectorXd eulerAngles = end_effector_state.rotation().eulerAngles(0,1,2);
        		Eigen::VectorXd eefPose(7);//Eigen::VectorXd eefPose(3);//
        		eefPose<<end_effector_state.translation(),quaternion.x(), quaternion.y(), quaternion.z(),quaternion.w();//, eulerAngles;//, quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();
                		//Eigen::VectorXd returnVec(4);
                		//returnVec<<eefPose.tail(4);
        		return eefPose;
		}	

		/* \brief Return Jacobian as 6xn matrix */
                Eigen::MatrixXd getJacobian(Eigen::VectorXd jointState)
                {
			//set joint position to input state
        		kinematic_state->setJointGroupPositions(joint_model_group, jointState);
        		//offset from the base frame - leave at 0,0,0
        		Eigen::Vector3d reference_point_position(0.0,0.0,0.0);
        		Eigen::MatrixXd jacobian;
        		kinematic_state->getJacobian(joint_model_group, kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back()),
                                     reference_point_position,
                                     jacobian, false /*true - Quaternion representation*/);
        		return jacobian.block(0,0,3,jacobian.cols());
	
                }

		double randomInRange( double _min, double _max ) {

    			return _min + ((_max - _min) * ((double)rand() / ((double)RAND_MAX + 1)));
		}	

	private:
		/* \brief Pointer to group for planning */
		robot_state::JointModelGroup* joint_model_group;
		/* \brief Pointer to current state of robot */
		robot_state::RobotStatePtr kinematic_state;

		robot_model::RobotModelPtr kinematic_model;

};

#endif

