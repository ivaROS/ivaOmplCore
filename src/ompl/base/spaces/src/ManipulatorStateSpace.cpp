/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
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
*   * Neither the name of the Rice University nor the names of its
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
	In this file are the definitions for the Manipulator State Space which
	provide functions like distance and interpolation for ManipulatorStates
	These states are defined by a joint state and a corresponding
	end-effector configuration
*/

#include "ompl/base/spaces/ManipulatorStateSpace.h"
//#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/tools/config/MagicConstants.h"

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm> 

#include <Eigen/Dense>
#include <Eigen/Geometry>  //tranpose
#include <math.h> 
#include <unsupported/Eigen/MatrixFunctions>  //matrix log

/* \brief Allocates a ManipulatorStateSpace state */
ompl::base::State* ompl::base::ManipulatorStateSpace::allocState(void) const
{
    StateType *state = new StateType();
    allocStateComponents(state);

    state->jointConfigKnown = new bool;
    state->eefPoseKnown = new bool;

    state->jointConfigKnown = false;
    state->eefPoseKnown = false;

    return state;
}

/* \brief Frees memory associated with state. May want to double check this */
void ompl::base::ManipulatorStateSpace::freeState(State *state) const
{
    CompoundStateSpace::freeState(state);
}

/* \brief Allocate state sampler for ManipulatorStateSpace */
ompl::base::StateSamplerPtr ompl::base::ManipulatorStateSpace::allocDefaultStateSampler(void) const
{
     std::cout<<"enter manipulator allocDefaultStateSampler"<<std::endl;
     ManipulatorStateSampler *ss = new ManipulatorStateSampler(this);
     return StateSamplerPtr(ss);
}

/* \brief Samples state uniformly - simply random between lower and upper bounds of joints */
void ompl::base::ManipulatorStateSampler::sampleUniform(State *state)
{
    const unsigned int dim = space_->getDimension();
    const RealVectorBounds &bounds = static_cast<const ManipulatorStateSpace*>(space_)->getBounds();
 
    ManipulatorStateSpace::StateType *rstate = static_cast<ManipulatorStateSpace::StateType*>(state);
    for (unsigned int i = 0 ; i < dim ; ++i)
        rstate->setJoints(rng_.uniformReal(bounds.low[i], bounds.high[i]), i);

    //Set state bools correctly
    rstate->jointConfigKnown = true;
    rstate->eefPoseKnown = false;

    	//std::cout<<"Sampled state: "<<std::endl;

    	//space_->printState(rstate, std::cout);
}

/* \brief Samples state uniformly near input state */
void ompl::base::ManipulatorStateSampler::sampleUniformNear(State *state, const State *near, const double distance)
{
    std::cout<<"enter manipulator state sampler: sampleUniformNear"<<std::endl;
    const unsigned int dim = space_->getDimension(); 
    const RealVectorBounds &bounds = static_cast<const ManipulatorStateSpace*>(space_)->getBounds();
 
    ManipulatorStateSpace::StateType *rstate = static_cast<ManipulatorStateSpace::StateType*>(state);
    const ManipulatorStateSpace::StateType *rnear = static_cast<const ManipulatorStateSpace::StateType*>(near);
    for (unsigned int i = 0 ; i < dim ; ++i)
    {
	double v = rng_.uniformReal(std::max(bounds.low[i], rnear->getJoints(i) - distance),
                             std::min(bounds.high[i], rnear->getJoints(i) + distance));
        rstate->setJoints(v, i);
    }
    rstate->jointConfigKnown = true;
}

/* \brief Samples state in a gaussian manner according to input parameters */
void ompl::base::ManipulatorStateSampler::sampleGaussian(State *state, const State *mean, const double stdDev)
{
    const unsigned int dim = space_->getDimension();
    const RealVectorBounds &bounds = static_cast<const ManipulatorStateSpace*>(space_)->getBounds();
 
    ManipulatorStateSpace::StateType *rstate = static_cast<ManipulatorStateSpace::StateType*>(state);
    const ManipulatorStateSpace::StateType *rmean = static_cast<const ManipulatorStateSpace::StateType*>(mean);
    for (unsigned int i = 0 ; i < dim ; ++i)
    {
        double v = rng_.gaussian(rmean->getJoints(i), stdDev);
        if (v < bounds.low[i])
            v = bounds.low[i];
        else
            if (v > bounds.high[i])
                v = bounds.high[i];
        rstate->setJoints(v, i);
    }
    rstate->jointConfigKnown = true;
}

//Some setup functions are called
void ompl::base::ManipulatorStateSpace::setup(void)
{
    getBounds().check();
    StateSpace::setup();
}

/* \brief Copy a given state into a newly allocated state */
void ompl::base::ManipulatorStateSpace::copyState(State *destination, const State *source) const
{
    CompoundStateSpace::copyState(destination, source);
   
    memcpy(&(static_cast<StateType*>(destination)->jointConfigKnown),
           &(static_cast<const StateType*>(source)->jointConfigKnown), sizeof(bool));

    memcpy(&(static_cast<StateType*>(destination)->eefPoseKnown),
           &(static_cast<const StateType*>(source)->eefPoseKnown), sizeof(bool));
}

/** \brief Set joint state of manipulator */
void ompl::base::ManipulatorStateSpace::setJoints(State *state, Eigen::VectorXd desiredJointValues) const
{
    //std::cout<<"enter setJoints(State *state, Eigen::VectorXd desiredJointValues)"<<std::endl;
    StateType *rstate = static_cast<StateType*>(state);
    //std::cout<<"cast state type"<<std::endl;
    if(desiredJointValues.size() != manipulatorDimension_)
    {
        //std::cout<<"size of desiredJointValues is not equal to manipulatorDimension_"<<std::endl;
	    OMPL_ERROR("Desired joint vector different than joint state size");
    }

    for (int i = 0; i < desiredJointValues.size(); ++i)
    {
        //std::cout<<"setJoints of specific state type"<<std::endl;
	    rstate->setJoints(desiredJointValues[i], i);
    }

    rstate->jointConfigKnown = true;
}

double ompl::base::ManipulatorStateSpace::getMaximumExtent(void) const
{
    double e = 0.0;
    for (unsigned int i = 0 ; i < manipulatorDimension_ ; ++i)
    {
        double d = getBounds().high[i] - getBounds().low[i];
        e += d*d;
    }
    return sqrt(e);
}

/* \brief Check whether state is within bounds */
bool ompl::base::ManipulatorStateSpace::satisfiesBounds(const State *state) const
{
    std::cout<<"ManipulatorStateSpace::satisfiesBounds"<<std::endl;
    const StateType *rstate = static_cast<const StateType*>(state);
    for (unsigned int i = 0 ; i < manipulatorDimension_ ; ++i)
    {
    std::cout<<"rstate->getJoints(i)"<<rstate->getJoints(i)<<std::endl;
	if (std::isnan(rstate->getJoints(i)))
	{
	    return false;
 	}
        else if (rstate->getJoints(i) - std::numeric_limits<double>::epsilon() > getBounds().high[i] ||
            rstate->getJoints(i) + std::numeric_limits<double>::epsilon() < getBounds().low[i])
            return false;
    }
    return true;
}

/** \brief Set joint state of manipulator */
void ompl::base::ManipulatorStateSpace::setJoints(State *state, std::vector<double> desiredJointValues) const
{
    StateType *rstate = static_cast<StateType*>(state);
    if(desiredJointValues.size() != manipulatorDimension_)
    {
	    OMPL_ERROR("Desired joint vector different than joint state size");
    }

    for (unsigned int i = 0; i < desiredJointValues.size(); ++i)
    {
	    rstate->setJoints(desiredJointValues[i], i);
    }

    rstate->jointConfigKnown = true;
}

/** \brief Get total joint state of manipulator */
std::vector<double> ompl::base::ManipulatorStateSpace::getJoints(const State *state) const
{

   const StateType *rstate = static_cast<const StateType*>(state);

   //RealVectorStateSpace vecSpace(manipulatorDimension_);
   std::vector<double> jointState;
   for (unsigned int i = 0; i < manipulatorDimension_; ++i)
   {
       //std::cout<<"get joint state joints"<<std::endl;
       //jointState.push_back(vecSpace.getJoints(rstate, i));
       jointState.push_back(rstate->getJoints(i));
       //std::cout<<"index: "<<i<<" finish pushing back"<<std::endl;
   }
   
   return jointState;
}

/** \brief Get total joint state of manipulator as eigen vector*/
Eigen::VectorXd ompl::base::ManipulatorStateSpace::getEigenJoints(const State *state) const
{

   const StateType *rstate = static_cast<const StateType*>(state);

   Eigen::VectorXd jointState(manipulatorDimension_);
   for (unsigned int i = 0; i < manipulatorDimension_; ++i)
   {
        jointState(i) = rstate->getJoints(i);
   }

   return jointState;
}


/** \brief Calculate and store state end-effector pose through forward kinematics. Modify given state */
void ompl::base::ManipulatorStateSpace::calculateEefPose(State *state)
{
    StateType* rstate = static_cast<StateType*>(state);

    SE3StateSpace::StateType* eefPose = &(rstate->endEffectorPose());
    Eigen::VectorXd poseVector = manipulatorState_->getEefPose(getEigenJoints(rstate));
    eefPose->setXYZ(poseVector(0), poseVector(1), poseVector(2));

    ///SO3StateSpace::StateType* eefRotation = &(eefPose->rotation());

    //eefRotation->w = poseVector(3); 
    //eefRotation->x = poseVector(4);
    //eefRotation->y = poseVector(5);
    //eefRotation->z = poseVector(6);

    rstate->eefPoseKnown = true;
}

/** \brief Calculate and store state end-effector pose through forward kinematics. Return as new SE3 State */
Eigen::VectorXd ompl::base::ManipulatorStateSpace::getEefPose(const State *state) const
{
    const StateType* rstate = static_cast<const StateType*>(state);

    Eigen::VectorXd poseVector = manipulatorState_->getEefPose(getEigenJoints(rstate));
 
    return poseVector;
}

/** \brief Return the Jacobian of the current state as an Eigen::Matrix */
Eigen::MatrixXd ompl::base::ManipulatorStateSpace::getJacobian(const State *state) const
{
    const StateType *rstate = static_cast<const StateType*>(state);
	
    return manipulatorState_->getJacobian(getEigenJoints(rstate));
}

/* \brief Compute distance between 2 states */
double ompl::base::ManipulatorStateSpace::distance(const State *state1, const State *state2) const
{
    /*
	2 possibilities
	1) Both states' joint space coordinate are known
	2) At least one of the states' joint space coordinate not known
    */

    const StateType* mState1 = static_cast<const StateType*>(state1);
    const StateType* mState2 = static_cast<const StateType*>(state2);
    double dist = 0.0;
    double dist1 = 0.0;
    double dist2 = 0.0;

    //std::cout<<"State 1 for distance: "<<std::endl;
    //printState(mState1, std::cout);
    //std::cout<<"State 2 for distance: "<<std::endl;
    //printState(mState2, std::cout);

/*
        //Possibility 1 - Real vector distance
    if (mState1->jointConfigKnown && mState2->jointConfigKnown)
    {
    const RealVectorStateSpace::StateType* jointConfig1 = &(mState1->jointConfig());
    const RealVectorStateSpace::StateType* jointConfig2 = &(mState2->jointConfig());
    
    RealVectorStateSpace vecSpace(manipulatorDimension_);
    dist = vecSpace.distance(jointConfig1, jointConfig2);    
    }

    //Possibility 2 - SE3 distance
    else
    {   
    assert(mState1->eefPoseKnown || mState2->eefPoseKnown);
    Eigen::VectorXd pose1(3);
    Eigen::VectorXd pose2(3);

    if (!mState1->eefPoseKnown)
    {
        pose1 = getEefPose(mState1);
        std::cout<<"position of from_state(x,y,z):"<<pose1(0)<<pose1(1)<<pose1(2)<<std::endl;
        const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
        pose2 << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();//,
        std::cout<<"position of to_state(x,y,z):"<<eefPose2->getX()<<eefPose2->getY()<<eefPose2->getZ()<<std::endl;
                         //eefPose2->rotation().w, eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z;
    }
    else if (!mState2->eefPoseKnown)
    {
        const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
        pose1 << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();//,
                         //eefPose1->rotation().w, eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z;
        pose2 = getEefPose(mState2);
    }
    else
    {
        const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
                pose1 << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();//,
                         //eefPose1->rotation().w, eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z;

        const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
        pose2 << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();//,
                         //eefPose2->rotation().w, eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z;

    }

    dist = (pose2-pose1).norm();
    }
std::cout<<"dist(position):"<<dist<<std::endl;
return dist;
*/


    //Possibility 1 - Real vector distance
    if (mState1->jointConfigKnown && mState2->jointConfigKnown)
    {
        //std::cout<<"mState1 and mState2's joint configurations are known"<<std::endl;
    	const RealVectorStateSpace::StateType* jointConfig1 = &(mState1->jointConfig());
    	const RealVectorStateSpace::StateType* jointConfig2 = &(mState2->jointConfig());
	
    	RealVectorStateSpace vecSpace(manipulatorDimension_);
    	dist = vecSpace.distance(jointConfig1, jointConfig2);   
        return dist; 
    }

    //Possibility 2 - SE3 distance
    else
    {	
    	assert(mState1->eefPoseKnown || mState2->eefPoseKnown);
    	Eigen::VectorXd pose1_position(3);
        //Eigen::Quaterniond pose1_orientation;
        Eigen::VectorXd pose1_orientation(4);
    	Eigen::VectorXd pose2_position(3);
        //Eigen::Quaterniond pose2_orientation;
        Eigen::VectorXd pose2_orientation(4);

    	if (!mState1->eefPoseKnown)
    	{
            //std::cout<<"mState1's eef configurations are unknown"<<std::endl;
            Eigen::VectorXd temp(7);
    		temp = getEefPose(mState1);
            pose1_position << temp(0), temp(1), temp(2);
            pose1_orientation << temp(3), temp(4), temp(5),temp(6);
            //pose1_orientation.x() = temp(3);
            //pose1_orientation.y() = temp(4);
            //pose1_orientation.z() = temp(5);
            //pose1_orientation.w() = temp(6);

            //std::cout<<"position of from_state(x,y,z):"<<temp(0)<<" "<<temp(1)<<" "<<temp(2)<<" "<<std::endl;
            //std::cout<<"orientation of from_state(x,y,z,w):"<<temp(3)<<" "<<temp(4)<<" "<<temp(5)<<" "<<temp(6)<<std::endl;

    		const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
    		pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose2->rotation().w;
            //pose2_orientation.x() = eefPose2->rotation().x;
            //pose2_orientation.y() = eefPose2->rotation().y;
            //pose2_orientation.z() = eefPose2->rotation().z;
            //pose2_orientation.w() = eefPose2->rotation().w;

            //std::cout<<"position of to_state(x,y,z):"<<eefPose2->getX()<<eefPose2->getY()<<eefPose2->getZ()<<std::endl;
            //std::cout<<"orientation of to_state(x,y,z,w):"<<eefPose2->rotation().x<<eefPose2->rotation().y<<eefPose2->rotation().z<<eefPose2->rotation().w<<std::endl;
    	}
    	else if (!mState2->eefPoseKnown)
    	{
            //std::cout<<"mState2's eef configurations are unknown"<<std::endl;
            Eigen::VectorXd temp(7);
    		const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
    		pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
            //pose1_orientation.x() = eefPose1->rotation().x;
            //pose1_orientation.y() = eefPose1->rotation().y;
            //pose1_orientation.z() = eefPose1->rotation().z;
            //pose1_orientation.w() = eefPose1->rotation().w;

    		temp = getEefPose(mState2);
            pose2_position << temp(0), temp(1), temp(2);
            pose2_orientation << temp(3), temp(4), temp(5), temp(6);
            //pose2_orientation.x() = temp(3);
            //pose2_orientation.y() = temp(4);
            //pose2_orientation.z() = temp(5);
            //pose2_orientation.w() = temp(6);
    	}
    	else
    	{
            //std::cout<<"mState1 and mState2's eef configurations are unknown"<<std::endl;
    		const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
            pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
            //pose1_orientation.x() = eefPose1->rotation().x;
            //pose1_orientation.y() = eefPose1->rotation().y;
            //pose1_orientation.z() = eefPose1->rotation().z;
            //pose1_orientation.w() = eefPose1->rotation().w;

    		const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
    		pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose1->rotation().w;
            //pose2_orientation.x() = eefPose2->rotation().x;
            //pose2_orientation.y() = eefPose2->rotation().y;
            //pose2_orientation.z() = eefPose2->rotation().z;
            //pose2_orientation.w() = eefPose2->rotation().w;
    	}

    	dist1 = (pose2_position-pose1_position).norm();
        
        /*          Method 1  (from OMPL)       */
        dist2 = fabs(pose1_orientation.x() * pose2_orientation.x() + pose1_orientation.y() * pose2_orientation.y() + pose1_orientation.z() * pose2_orientation.z() + pose1_orientation.w() * pose2_orientation.w());
        if (dist2 > 1.0 - 1e-9)
            dist2 = 0.0;
        else
            dist2 = acos(dist2);
        
        
        /*Method 2 for computing distance of orientation*/
        //Eigen::MatrixXd R1 = pose1_orientation.normalized().toRotationMatrix();
        //Eigen::MatrixXd R2 = pose2_orientation.normalized().toRotationMatrix();
        //Eigen::MatrixXd R = (R1 * R2.transpose()).log();
        //for (int i = 0; i < R.rows(); i++)
        //    for(int j = 0; j < R.cols(); j++)
        //       dist2 = dist2 + R(i,j) * R(i,j); 
        //dist2 = sqrt(dist2);
        

        /*Method 3 for computing distance of orientation*/
        //dist2 = std::min((pose2_orientation-pose1_orientation).norm(),(pose2_orientation+pose1_orientation).norm());


        /* Method 4 for computing distance of orientation*/
        //dist2 = 1.0 - fabs(pose1_orientation.transpose() * pose2_orientation);


        /*     Metrices for SE(3)    */
        /*
        Eigen::VectorXd pose1_position(3);
        Eigen::Quaterniond pose1_orientation;
        Eigen::VectorXd pose2_position(3);
        Eigen::Quaterniond pose2_orientation;
        double q10,q11,q12,q13,q20,q21,q22,q23;

        if (!mState1->eefPoseKnown)
        {
            //state1
            Eigen::VectorXd temp(7);
            temp = getEefPose(mState1);
            pose1_position << temp(0), temp(1), temp(2);
            q11 = temp(3);
            q12 = temp(4);
            q13 = temp(5);
            q10 = temp(6);
            //state2
            const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
            pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            q21 = eefPose2->rotation().x;
            q22 = eefPose2->rotation().y;
            q23 = eefPose2->rotation().z;
            q20 = eefPose2->rotation().w;
        }
        else if (!mState2->eefPoseKnown)
        {
            //state1
            Eigen::VectorXd temp(7);
            const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
            pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            q11 = eefPose1->rotation().x;
            q12 = eefPose1->rotation().y;
            q13 = eefPose1->rotation().z;
            q10 = eefPose1->rotation().w;
            //state2
            temp = getEefPose(mState2);
            pose2_position << temp(0), temp(1), temp(2);
            q21 = temp(3);
            q22 = temp(4);
            q23 = temp(5);
            q20 = temp(6);
        }
        else
        {
            //state1
            const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
            pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            q11 = eefPose1->rotation().x;
            q12 = eefPose1->rotation().y;
            q13 = eefPose1->rotation().z;
            q10 = eefPose1->rotation().w;
            //state2
            const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
            pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            q21 = eefPose2->rotation().x;
            q22 = eefPose2->rotation().y;
            q23 = eefPose2->rotation().z;
            q20 = eefPose2->rotation().w;
        }
        //construct transformation matrix for state1 and state2
        Eigen::Matrix4d state1PoseG = Eigen::MatrixXd::Identity(4,4);
        Eigen::Matrix4d state2PoseG = Eigen::MatrixXd::Identity(4,4);
        Eigen::Matrix3d state1Rot, state2Rot;

        state1PoseG(0,0) = 1.0 - 2.0 * (q12 * q12 + q13 * q13);
        state1PoseG(0,1) = 2.0 * (q11 * q12 - q10 * q13);
        state1PoseG(0,2) = 2.0 * (q10 * q12 + q11 * q13);
        state1PoseG(1,0) = 2.0 * (q11 * q12 + q10 * q13);
        state1PoseG(1,1) = 1.0 - 2.0 * (q11 * q11 + q13 * q13);
        state1PoseG(1,2) = 2.0 * (q12 * q13 - q10 * q11);
        state1PoseG(2,0) = 2.0 * (q11 * q13 - q10 * q12);
        state1PoseG(2,1) = 2.0 * (q10 * q11 + q12 * q13);
        state1PoseG(2,2) = 1.0 - 2.0 * (q11 * q11 + q12 * q12);
        state1PoseG.block(0,3,3,1) = pose1_position.block(0,0,3,1);
        state1Rot << state1PoseG.block(0,0,3,3);

        state2PoseG(0,0) = 1.0 - 2.0 * (q22 * q22 + q23 * q23);
        state2PoseG(0,1) = 2.0 * (q21 * q22 - q20 * q23);
        state2PoseG(0,2) = 2.0 * (q20 * q22 + q21 * q23);
        state2PoseG(1,0) = 2.0 * (q21 * q22 + q20 * q23);
        state2PoseG(1,1) = 1.0 - 2.0 * (q21 * q21 + q23 * q23);
        state2PoseG(1,2) = 2.0 * (q22 * q23 - q20 * q21);
        state2PoseG(2,0) = 2.0 * (q21 * q23 - q20 * q22);
        state2PoseG(2,1) = 2.0 * (q20 * q21 + q22 * q23);
        state2PoseG(2,2) = 1.0 - 2.0 * (q21 * q21 + q22 * q22);
        state2PoseG.block(0,3,3,1) = pose2_position.block(0,0,3,1);
        state2Rot << state2PoseG.block(0,0,3,3);

        //compute the twists for state1 and state2
        Eigen::Matrix3d w1_head, w2_head;
        Eigen::Vector3d w1,w2,v1,v2;
        Eigen::Matrix3d eye = Eigen::MatrixXd::Identity(3,3);
        double theta1,theta2;
        theta1 = acos(0.5 * (state1PoseG(0,0) + state1PoseG(1,1) + state1PoseG(2,2) - 1.0));
        theta2 = acos(0.5 * (state2PoseG(0,0) + state2PoseG(1,1) + state2PoseG(2,2) - 1.0));
        w1 << (state1PoseG(2,1) - state1PoseG(1,2)) / (2.0 * sin(theta1)), (state1PoseG(0,2) - state1PoseG(2,0)) / (2.0 * sin(theta1)), (state1PoseG(1,0) - state1PoseG(0,1)) / (2.0 * sin(theta1));
        w2 << (state2PoseG(2,1) - state2PoseG(1,2)) / (2.0 * sin(theta2)), (state2PoseG(0,2) - state2PoseG(2,0)) / (2.0 * sin(theta2)), (state2PoseG(1,0) - state2PoseG(0,1)) / (2.0 * sin(theta2));
        w1_head = (state1Rot - state1Rot.transpose()) / (2.0 * sin(theta1));
        w2_head = (state2Rot - state2Rot.transpose()) / (2.0 * sin(theta2));
        v1 = ((eye - state1Rot) * w1_head + theta1 * w1 * w1.transpose()).inverse() * pose1_position;
        v2 = ((eye - state2Rot) * w2_head + theta2 * w2 * w2.transpose()).inverse() * pose2_position;
        
        //compute the distance
        Eigen::Matrix3d trace;
        trace = w1_head.transpose() * w2_head;

        dist = trace(0,0) + trace(1,1) + trace(2,2) + v1.transpose() * v2;
    }
    std::cout<<"distance: "<<dist<<std::endl;
    return dist;
    */    
    
    }

    if(dist1 > 0.02)
        return dist1;
    else
        return dist2;
    

}

double ompl::base::ManipulatorStateSpace::getscore(const State *state1, const State *state2) const
{
    /*
    2 possibilities
    1) Both states' joint space coordinate are known
    2) At least one of the states' joint space coordinate not known
    */
    
    //just return position as inverse of score
    
    const StateType* mState1 = static_cast<const StateType*>(state1);
    const StateType* mState2 = static_cast<const StateType*>(state2);
    double dist = 0.0;
    double dist1 = 0.0;
    //double dist2 = 0.0;

    //Possibility 1 - Real vector distance
    if (mState1->jointConfigKnown && mState2->jointConfigKnown)
    {
        const RealVectorStateSpace::StateType* jointConfig1 = & (mState1->jointConfig()); 
        const RealVectorStateSpace::StateType* jointConfig2 = &(mState2->jointConfig());
    
        RealVectorStateSpace vecSpace(manipulatorDimension_);
        dist = vecSpace.distance(jointConfig1, jointConfig2);  
    }

    //Possibility 2 - SE3 distance
    else
    {   
        assert(mState1->eefPoseKnown || mState2->eefPoseKnown);
        Eigen::VectorXd pose1_position(3);
        Eigen::Quaterniond pose1_orientation;
        Eigen::VectorXd pose2_position(3);
        Eigen::Quaterniond pose2_orientation;

        if (!mState1->eefPoseKnown)
        {
            Eigen::VectorXd temp(7);
            temp = getEefPose(mState1);
            pose1_position << temp(0), temp(1), temp(2);
            //pose1_orientation << temp(3), temp(4), temp(5),temp(6);
            //pose1_orientation.x() = temp(3);
            //pose1_orientation.y() = temp(4);
            //pose1_orientation.z() = temp(5);
            //pose1_orientation.w() = temp(6);

            const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
            pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            //pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose2->rotation().w;
            //pose2_orientation.x() = eefPose2->rotation().x;
            //pose2_orientation.y() = eefPose2->rotation().y;
            //pose2_orientation.z() = eefPose2->rotation().z;
            //pose2_orientation.w() = eefPose2->rotation().w;
        }
        else if (!mState2->eefPoseKnown)
        {
            //std::cout<<"mState2's eef configurations are unknown"<<std::endl;
            Eigen::VectorXd temp(7);
            const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
            pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            //pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
            //pose1_orientation.x() = eefPose1->rotation().x;
            //pose1_orientation.y() = eefPose1->rotation().y;
            //pose1_orientation.z() = eefPose1->rotation().z;
            //pose1_orientation.w() = eefPose1->rotation().w;

            temp = getEefPose(mState2);
            pose2_position << temp(0), temp(1), temp(2);
            //pose2_orientation << temp(3), temp(4), temp(5), temp(6);
            //pose2_orientation.x() = temp(3);
            //pose2_orientation.y() = temp(4);
            //pose2_orientation.z() = temp(5);
            //pose2_orientation.w() = temp(6);
        }
        else
        {
            //std::cout<<"mState1 and mState2's eef configurations are unknown"<<std::endl;
            const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
            pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ();
            //pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
            //pose1_orientation.x() = eefPose1->rotation().x;
            //pose1_orientation.y() = eefPose1->rotation().y;
            //pose1_orientation.z() = eefPose1->rotation().z;
            //pose1_orientation.w() = eefPose1->rotation().w;

            const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
            pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
            //pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose1->rotation().w;
            //pose2_orientation.x() = eefPose2->rotation().x;
            //pose2_orientation.y() = eefPose2->rotation().y;
            //pose2_orientation.z() = eefPose2->rotation().z;
            //pose2_orientation.w() = eefPose2->rotation().w;
        }

        dist1 = (pose2_position-pose1_position).norm();
        //dist2 = fabs(pose1_orientation.x() * pose2_orientation.x() + pose1_orientation.y() * pose2_orientation.y() + pose1_orientation.z() * pose2_orientation.z() + pose1_orientation.w() * pose2_orientation.w());
        //if (dist2 > 1.0 - 1e-9)
        //    dist2 = 0.0;
        //else
        //    dist2 = acos(dist2);
        //dist = dist1 * 0.8 + dist2 * 0.2;
        dist = dist1;
    }
    
    return dist;
    

}

/* \brief Check whether two states are equal - depends on what is known about the states */
bool ompl::base::ManipulatorStateSpace::equalStates(const State *state1, const State *state2) const
{
    const StateType* s1 = static_cast<const StateType*>(state1);
    const StateType* s2 = static_cast<const StateType*>(state2);

    if (s1->jointConfigKnown && s2->jointConfigKnown)
    {
        for (unsigned int i = 0 ; i < manipulatorDimension_ ; ++i)
        {
           double diff = s1->getJoints(i) - s2->getJoints(i);
           if (fabs(diff) > std::numeric_limits<double>::epsilon() * 2.0)
               return false;
        }
    }

    else
    {
	Eigen::VectorXd pose1(3);
	Eigen::VectorXd pose2(3);

	if (!s1->eefPoseKnown)
        {
                pose1 = getEefPose(s1);
                const SE3StateSpace::StateType* eefPose2 = &(s2->endEffectorPose());
                pose2 << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ(),
                         eefPose2->rotation().w, eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z;
        }
        else if (!s2->eefPoseKnown)
        {
               const SE3StateSpace::StateType* eefPose1 = &(s1->endEffectorPose());
               pose1 << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ(),
                         eefPose1->rotation().w, eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z;
               pose2 = getEefPose(s2);
        }
	

	return ((pose2-pose1).norm())<(std::numeric_limits<double>::epsilon() * 2.0);
    }

return true;
}

/* Interpolate between two states by a factor of t */
void ompl::base::ManipulatorStateSpace::interpolate(const State *from, const State *to, const double t, State *state) const
{
    //std::cout<<"Interpolating in OMPL"<<std::endl;
/*
	2 Possibilities
	1)  "to" has known joint-space configuration
	2) State "to" does not have known joint-space configuration (use Jacobian)
*/
    std::cout<<"step size: "<<t<<std::endl;
    const StateType* mStateFrom = static_cast<const StateType*>(from);
    const StateType* mStateTo = static_cast<const StateType*>(to);
    StateType* mState = static_cast<StateType*>(state);

    	//std::cout<<"State from which we interpolate: "<<std::endl;
    	//printState(mStateFrom, std::cout);
    	//std::cout<<"State to which we interpolate: "<<std::endl;
    	//printState(mStateTo, std::cout);

    //State "from" must have defined joint configuration
    assert(mStateFrom->jointConfigKnown);

    //Possibility 1
    if( mStateTo->jointConfigKnown )
    {
        std::cout<<"tostate's joint configuration is known"<<std::endl;
	    RealVectorStateSpace vecSpace(manipulatorDimension_);
	    vecSpace.interpolate(&(mStateFrom->jointConfig()), &(mStateTo->jointConfig()), t, &(mState->jointConfig()));
	    mState->jointConfigKnown = true;
		//std::cout<<"Resultant state: "<<std::endl;
        	//printState(mState, std::cout);
    }

    //Possibility 2 - Jacobian interpolation
    else if (!mStateTo->jointConfigKnown)
    {
        std::cout<<"tostate's joint configuration is unknown"<<std::endl;
    /********Jacobian just position*******************/
    //version of Leo
    /*
        std::cout<<"tostate's joint configuration is unknown"<<std::endl;
	    assert(mStateTo->eefPoseKnown);

        Eigen::VectorXd temp1 = getEefPose(mStateFrom)
        Eigen::VectorXd poseFrom = temp(0,0,6,1);
	    Eigen::VectorXd poseTo(6);

        const SE3StateSpace::StateType* eefPoseTo = &(mStateTo->endEffectorPose());

	    Eigen::MatrixXd JacPseudoInv = getPseudoInvJacobian(mStateFrom);

		//Eigen::MatrixXd Jac = getJacobian(mStateFrom);
        	//std::cout<<"Jacobian"<<std::endl;
		//std::cout<<Jac<<std::endl;

		//Eigen::MatrixXd JacPseudoInv = getJacobian(mStateFrom).transpose();
		//std::cout<<"Pseudo Inverse"<<std::endl;
		//std::cout<<JacPseudoInv<<std::endl;
	    Eigen::VectorXd jointsFromEigen = getEigenJoints(mStateFrom);
		//std::cout<<"Initial joints: "<<jointsFromEigen<<std::endl;

		//std::cout<<"Calculated pose from: "<<poseFrom<<std::endl;

	    poseTo << eefPoseTo->getX(), eefPoseTo->getY(), eefPoseTo->getZ();//, eefPoseTo->rotation().w, 
	    //eefPoseTo->rotation().x, eefPoseTo->rotation().y, eefPoseTo->rotation().z;
		//std::cout<<"Calculated pose to: "<<poseTo<<std::endl;

	    Eigen::VectorXd diff = poseTo - poseFrom;
		//std::cout<<"Calculated pose to: "<<poseTo<<std::endl;
       	 	//std::cout<<"Calculated pose difference: "<<diff<<std::endl;
		//std::cout<<"<-----StepSize: "<<t<<" ----->"<<std::endl;
	    Eigen::VectorXd dJoints = JacPseudoInv*(diff.normalized())*t;

	    Eigen::VectorXd jointsInterp = jointsFromEigen + dJoints;
	    Eigen::VectorXd poseNew = manipulatorState_->getEefPose(jointsInterp);
		//std::cout<<"Calculated pose new: "<<poseNew<<std::endl;

	    //Eigen::VectorXd properVector = diff.normalized();
	    Eigen::VectorXd actualVectorDiff = poseNew-poseFrom;
	    Eigen::VectorXd actualVector = actualVectorDiff.normalized();
		//double dotProduct = properVector.dot(actualVector);
		//std::cout<<"<-----Dot product: "<<dotProduct<<" ----->"<<std::endl;
		//std::cin.get();
		//std::cout<<"Assigning to new state"<<std::endl;
	    setJoints(mState, jointsInterp);
		//std::cout<<"Done assigning."<<std::endl;
    */


    /**********Jacobian position and orientation*****************/
    //version of separate positiona and orientation by Ruinian Xu
    //variables
    
    Eigen::VectorXd initialJoints(7); // 7 by 1 vector //7J
    //Eigen::VectorXd initialPose(6); // 6 by 1 vector for twist
    Eigen::Matrix4d initialPoseG = Eigen::MatrixXd::Identity(4,4); // 4 by 4 matrix for transformation
    Eigen::VectorXd initialLocation(3); 
    
    Eigen::VectorXd finalLocation(3); 
    Eigen::MatrixXd finalPoseG = Eigen::MatrixXd::Identity(4,4);

    Eigen::Vector3d w(0.0,0.0,0.0);
    Eigen::Vector3d v(0.0,0.0,0.0);
    Eigen::MatrixXd twist_spatial = Eigen::MatrixXd::Zero(4,4);

    Eigen::VectorXd currentLocation(3);
    Eigen::VectorXd jointsVelocity;
    Eigen::VectorXd newJoints;
    Eigen::Matrix4d relativePoseG;
    Eigen::Matrix3d relativeR;
    Eigen::Vector3d relativeP(0.0,0.0,0.0);

    Eigen::MatrixXd jacobian_all = Eigen::MatrixXd::Zero(6,8); // 6 by 8 to catch Jacobian matrix from ROS
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6,7); // 6 by 7 Jacobian matrix (7 joints) //7J
    Eigen::MatrixXd jacPseudoInv; // 7 by 6 inverse Jacobian matrix (7 by 6 invJ* 6 by 1 twist = 7 by 1 new joint values )//7J

    Eigen::Vector3d reference_point_position(0.0,0.0,0.0);
    //Eigen::Matrix3d eyes = Eigen::MatrixXd::Identity(3,3);
    //Eigen::Matrix3d W_head;

    //construct rotation matrix for goal state
    const SE3StateSpace::StateType* eefPoseTo = &(mStateTo->endEffectorPose());
    double q11 = eefPoseTo->rotation().x;
    double q12 = eefPoseTo->rotation().y;
    double q13 = eefPoseTo->rotation().z;
    double q10 = eefPoseTo->rotation().w;
    Eigen::Matrix3d FinalRot;
    
    /*
    //compute the euler angle from quarternion
    double ax = std::atan2(2.0*(q10*q11+q12*q13),1.0-2.0*(q11*q11+q12*q12));
    double ay = std::asin(2.0*(q10*q12-q11*q13));
    double az = std::atan2(2.0*(q10*q13+q12*q11),1.0-2.0*(q13*q13+q12*q12));
    //construct x,y,z axis rotation matrix
    Eigen::Matrix3d RotX = Eigen::MatrixXd::Identity(3,3);
    Eigen::Matrix3d RotY = Eigen::MatrixXd::Identity(3,3);
    Eigen::Matrix3d RotZ = Eigen::MatrixXd::Identity(3,3);
    RotX(1,1) = std::cos(ax);
    RotX(1,2) = std::sin(-ax);
    RotX(2,2) = std::cos(ax);
    RotX(2,1) = std::sin(ax);

    RotY(0,0) = std::cos(ay);
    RotY(2,2) = std::cos(ay);
    RotY(0,2) = std::sin(ay);
    RotY(2,0) = std::sin(-ay);

    RotZ(0,0) = std::cos(az);
    RotZ(1,1) = std::cos(az);
    RotZ(1,0) = std::sin(az);
    RotZ(0,1) = std::sin(-az);

    FinalRot = RotX * RotY * RotZ;
    */
    
    FinalRot(0,0) = 1.0 - 2.0 * (q12 * q12 + q13 * q13);
    FinalRot(0,1) = 2.0 * (q11 * q12 - q10 * q13);
    FinalRot(0,2) = 2.0 * (q10 * q12 + q11 * q13);
    FinalRot(1,0) = 2.0 * (q11 * q12 + q10 * q13);
    FinalRot(1,1) = 1.0 - 2.0 * (q11 * q11 + q13 * q13);
    FinalRot(1,2) = 2.0 * (q12 * q13 - q10 * q11);
    FinalRot(2,0) = 2.0 * (q11 * q13 - q10 * q12);
    FinalRot(2,1) = 2.0 * (q10 * q11 + q12 * q13);
    FinalRot(2,2) = 1.0 - 2.0 * (q11 * q11 + q12 * q12);
    
    
    finalPoseG.block(0,0,3,3) << FinalRot;
    finalLocation << eefPoseTo->getX(), eefPoseTo->getY(), eefPoseTo->getZ();
    finalPoseG.block(0,3,3,1) = finalLocation.block(0,0,3,1);

    //assign current state joint values
    const RealVectorStateSpace::StateType* jointConfig = &(mStateFrom->jointConfig());
    std::vector<double> current_joint_values;
    for (int i = 0; i < 7; i++)
    {
        current_joint_values.push_back(jointConfig->values[i]);
    }

    //int count = 0;
    //double dist = distance(mStateFrom,mStateTo);

    //load robot model in order to get kinematic model
    /*
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));
    const robot_state::JointModelGroup* joint_model_group = kinematic_model->getJointModelGroup("arm4");
    */

    //while(count < 10 && dist > 0.03)
    //{

        //kinematic_state->setJointGroupPositions(joint_model_group, current_joint_values);

        //construct inital state information
        Eigen::VectorXd initial_pose;
        initial_pose = getEefPose(from);
        initialJoints << current_joint_values[0], current_joint_values[1], current_joint_values[2], current_joint_values[3], current_joint_values[4], current_joint_values[5], current_joint_values[6];
        initialLocation << initial_pose(0),initial_pose(1),initial_pose(2);
        double q21 = initial_pose(3);
        double q22 = initial_pose(4);
        double q23 = initial_pose(5);
        double q20 = initial_pose(6);
        //const Eigen::Affine3d &end_effector_state = kinematic_state->getGlobalLinkTransform("link_8");//kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back())
        //initialPose << end_effector_state.translation(), end_effector_state.rotation().eulerAngles(0,1,2);
        Eigen::Matrix3d InitialRot;
        /*
        double ax2 = std::atan2(2.0*(q20*q21+q22*q23),1.0-2.0*(q21*q21+q22*q22));
        double ay2 = std::asin(2.0*(q20*q22-q21*q23));
        double az2 = std::atan2(2.0*(q20*q23+q22*q21),1.0-2.0*(q23*q23+q22*q22));
        //construct x,y,z axis rotation matrix
        Eigen::Matrix3d RotX2 = Eigen::MatrixXd::Identity(3,3);
        Eigen::Matrix3d RotY2 = Eigen::MatrixXd::Identity(3,3);
        Eigen::Matrix3d RotZ2 = Eigen::MatrixXd::Identity(3,3);
        RotX2(1,1) = std::cos(ax2);
        RotX2(1,2) = std::sin(-ax2);
        RotX2(2,2) = std::cos(ax2);
        RotX2(2,1) = std::sin(ax2);

        RotY2(0,0) = std::cos(ay2);
        RotY2(2,2) = std::cos(ay2);
        RotY2(0,2) = std::sin(ay2);
        RotY2(2,0) = std::sin(-ay2);

        RotZ2(0,0) = std::cos(az2);
        RotZ2(1,1) = std::cos(az2);
        RotZ2(1,0) = std::sin(az2);
        RotZ2(0,1) = std::sin(-az2);

        InitialRot = RotX2 * RotY2 * RotZ2;
        */

        
        InitialRot(0,0) = 1.0 - 2.0 * (q22 * q22 + q23 * q23);
        InitialRot(0,1) = 2.0 * (q21 * q22 - q20 * q23);
        InitialRot(0,2) = 2.0 * (q20 * q22 + q21 * q23);
        InitialRot(1,0) = 2.0 * (q21 * q22 + q20 * q23);
        InitialRot(1,1) = 1.0 - 2.0 * (q21 * q21 + q23 * q23);
        InitialRot(1,2) = 2.0 * (q22 * q23 - q20 * q21);
        InitialRot(2,0) = 2.0 * (q21 * q23 - q20 * q22);
        InitialRot(2,1) = 2.0 * (q20 * q21 + q22 * q23);
        InitialRot(2,2) = 1.0 - 2.0 * (q21 * q21 + q22 * q22);
        
        initialPoseG.block(0,0,3,3) << InitialRot;
        initialPoseG.block(0,3,3,1) = initialLocation.block(0,0,3,1);
        //initialPoseG = end_effector_state.matrix();

        //Joint values to inverse Jacobian               
        // Get the Jacobian
        //kinematic_state->getJacobian(joint_model_group, kinematic_state->getLinkModel(joint_model_group->getLinkModelNames()[6]), reference_point_position, jacobian_all);//7J
                
        // Get inverse Jacobian of 7 links
        jacPseudoInv = getPseudoInvJacobian(from);

        // Relative Pose to Twist                      
        // get relativePoseG as spatial frame
        /*
        relativePoseG = finalPoseG * initialPoseG.inverse();
        relativeR = relativePoseG.block(0,0,3,3);
        relativeP = relativePoseG.block(0,3,3,1);
        double traceR = relativeR(0,0) + relativeR(1,1) + relativeR(2,2);

        // calculate v, w depending on tau
        double tau = acos(0.5*(traceR - 1));
        if (tau < 0.1)
        {
            w << 0.0,0.0,0.0;
            v << relativeP;
        }
        else
        {
            W_head = (relativeR - relativeR.transpose()) / (2*sin(tau));
            w << W_head(2,1), W_head(0,2), W_head(1,0);
             
            v = ((eyes - relativeR) * W_head + w * w.transpose() * tau).inverse() * relativeP;
        }

        // get twist
        double vnormScale;
        double wnormScale;

        if ((finalLocation - initialLocation).norm() > 0.15)
        {
            vnormScale = 5;
            wnormScale = 5;
        }
        else if ((finalLocation - initialLocation).norm() > 0.05)
        {
            vnormScale = 10;
            wnormScale = 10;
        }
        else
        {
            vnormScale = 20;
            wnormScale = 20;
        }
        */

        /*
        v = v / (vnormScale * v.norm());
        if(w.norm() != 0) 
            w = w / (wnormScale * w.norm());
        else 
            w << 0.0, 0.0, 0.0;
        */
        
        //compute hybrid velocity by first using spatial velocity and transform it to body and then hybrid
        /*
        twist_spatial.block(0,0,3,3) << W_head;
        twist_spatial.block(0,3,3,1) << v;

        Eigen::MatrixXd twist_body;
        twist_body = initialPoseG.inverse() * twist_spatial * initialPoseG;
        Eigen::VectorXd twist(6);
        twist << twist_body(0,3),twist_body(1,3),twist_body(2,3),twist_body(2,1),twist_body(0,2),twist_body(1,0);
        //transformation matrix from body to hybrid
        Eigen::MatrixXd tf = Eigen::MatrixXd::Zero(6,6);
        tf.block(0,0,3,3) << InitialRot;
        tf.block(3,3,3,3) << InitialRot;
        Eigen::VectorXd hybrid_velocity;
        hybrid_velocity = tf * twist;
        */

        //compute hybrid velocity by separately computing w and v
        Eigen::MatrixXd relativeRot = InitialRot.inverse() * FinalRot;
        Eigen::MatrixXd Rot_body_hat = relativeRot.log();
        //std::cout<<"Rot_body_head: \n"<<Rot_body_hat<<std::endl;
        Eigen::VectorXd Rot_body(3);
        Rot_body << Rot_body_hat(2,1), Rot_body_hat(0,2), Rot_body_hat(1,0);
        Eigen::VectorXd Rot_spatial(3);
        Rot_spatial = InitialRot * Rot_body;
        //std::cout<<"Rot_spatial: \n"<<Rot_spatial<<std::endl;

        Eigen::VectorXd Trans_body = finalLocation - initialLocation;
        Trans_body = InitialRot.inverse() * Trans_body;
        Eigen::VectorXd temp1 = InitialRot * Rot_body;//Ri * wb
        Eigen::VectorXd temp2(3);
        temp2 << (initialLocation(1) * temp1(2) - temp1(1) * initialLocation(2)), (initialLocation(2) * temp1(0) - temp1(2) * initialLocation(0)), (initialLocation(0) * temp1(1) - temp1(0) * initialLocation(1));
        Eigen::VectorXd Trans_spatial =  InitialRot * (Trans_body.normalized() * t) + temp2;
        Trans_spatial = Trans_spatial;

        Eigen::VectorXd spatial_velocity(6);
        spatial_velocity.block(0,0,3,1) = Trans_spatial;
        spatial_velocity.block(3,0,3,1) = Rot_spatial * t;

        //Update Joint values                       
        jointsVelocity = jacPseudoInv * spatial_velocity;
        newJoints = initialJoints + jacPseudoInv * spatial_velocity;

        //std::cout<<"newJoints"<<std::endl;
        //std::cout<<newJoints<<std::endl;
        //std::cout<<"twist"<<std::endl;
        //std::cout<<twist<<std::endl;

        //update the current joints value
        for (int i = 0; i < 7; i++)
        {
            current_joint_values[i] = newJoints(i);
        }

        //check whether joints are over limit
        for (unsigned int i = 0; i < manipulatorDimension_; i++)
        {
            if (newJoints(i) > 1.8)
                newJoints(i) = 1.8;
            else if (newJoints(i) < -1.8)
                newJoints(i) = -1.8;
        }

        //update the distance between the current interpreted state and goal state
        //set joint configuration for interpreted state
        setJoints(mState, newJoints);
        //dist = distance(mState,mStateTo);

        //if (std::isnan(dist))
        //    break;
            
        //std::cout<<"During Jacobian, distance for loop "<<count<<" is "<<dist<<std::endl;
        //update the counter
        //count++;
    //}
    }
}

/* \brief Return the Jacobian pseudo-inverse of the manipulator instance of the class */
Eigen::MatrixXd ompl::base::ManipulatorStateSpace::getPseudoInvJacobian(const State *inputState) const
{

    const StateType* mInputState = static_cast<const StateType*>(inputState);

    Eigen::MatrixXd Jaclin = manipulatorState_->getJacobian(getEigenJoints(mInputState));

    /*
    Eigen::MatrixXd JaclinT = Jaclin.transpose();
    Eigen::MatrixXd Jt;
    Eigen::MatrixXd JJt = (Jaclin*JaclinT);
    Eigen::FullPivLU<Eigen::MatrixXd> lu(JJt);
    Jt = JaclinT*( lu.inverse() );

    return Jt;
    */

    Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(6,6);//7J 
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jaclin, Eigen::ComputeThinU | Eigen::ComputeThinV);
    for(size_t idx = 0; idx < 6; idx++)
    {// only 6 no matter link_7 or link_8
        double singularvalue = svd.singularValues()[idx];
        if(singularvalue < 0.03) 
        {
            S_inv(idx,idx) = 0.0;
            std::cout<<"singularity occurred!"<<std::endl;
        }
        else 
            S_inv(idx,idx) = 1.0 / singularvalue;
    }
    Eigen::MatrixXd tempV = Eigen::MatrixXd::Ones(7,6);//7J
    tempV.block(0,0,7,6) = svd.matrixV();

    Eigen::MatrixXd jacPseudoInv = tempV * S_inv * svd.matrixU().inverse();

    return jacPseudoInv;
}

/* \brief Return the Euler angles corresponding to an input SO3 object (stored as unit quaternion) */
std::vector<double> ompl::base::ManipulatorStateSpace::getEulerAngles(const SO3StateSpace::StateType* inputSO3) const
{
    double x = inputSO3->x;
    double y = inputSO3->y;	
    double z = inputSO3->z;
    double w = inputSO3->w;

    double rX = atan2(2*(x*y + z*w), 1-2*(y*y+z*z));
    double rY = asin(2*(x*z-w*y));
    double rZ = atan2(2*(x*w + y*z), 1-2*(z*z+w*w));

    std::vector<double> eulerAngles;
    eulerAngles.push_back(rX);
    eulerAngles.push_back(rY);
    eulerAngles.push_back(rZ);

    return eulerAngles; //Check this
}

/* \brief Return the unit quaternion representation for a given set of Euler angles */
Eigen::VectorXd ompl::base::ManipulatorStateSpace::getQuaternion(double rX, double rY, double rZ) const
{
    double x = cos(rX/2)*cos(rY/2)*cos(rZ/2) + sin(rX/2)*sin(rY/2)*sin(rZ/2);
    double y = sin(rX/2)*cos(rY/2)*cos(rZ/2) - cos(rX/2)*sin(rY/2)*sin(rZ/2);
    double z = cos(rX/2)*sin(rY/2)*cos(rZ/2) + sin(rX/2)*cos(rY/2)*sin(rZ/2);
    double w = cos(rX/2)*cos(rY/2)*sin(rZ/2) - sin(rX/2)*sin(rY/2)*cos(rZ/2);

    Eigen::VectorXd quaternion(4);
    quaternion << x, y, z, w;

    return quaternion; //Check this
}

/* \brief Display the Manipulator State */
void ompl::base::ManipulatorStateSpace::printState(const State *state, std::ostream &out) const
{
    std::cout<<"enter manipulator state space printState"<<std::endl;
    out << "Manipulator state [" << std::endl;
    const StateType *mstate = static_cast<const StateType*>(state);
    //Joint config
    out << "Joint configuration [ ";
    if (mstate->jointConfigKnown)
    {
        std::cout<<"joint configuration is known"<<std::endl;
        for (unsigned int i = 0; i < manipulatorDimension_; ++i)
             out << mstate->getJoints(i) << " ";
        out << "]" << std::endl;
    }
    else
    {
        for (unsigned int i = 0; i < manipulatorDimension_; ++i)
              out << "X" << " ";
        out << "]" << std::endl;
    }

    //SE3 pose
    out << "End-effector pose [" << std::endl;
    if (mstate->eefPoseKnown)
    {
        std::cout<<"pose configuration is known"<<std::endl;
        out << "Translation [ " << mstate->endEffectorPose().getX() << " "
            << mstate->endEffectorPose().getY() << " " << mstate->endEffectorPose().getZ() << " ]";
        out << std::endl;

        out << "Rotation [ " << mstate->endEffectorPose().rotation().x << " " << mstate->endEffectorPose().rotation().y << " "
            << mstate->endEffectorPose().rotation().z << " " << mstate->endEffectorPose().rotation().w << " ]";
        out << std::endl;
    }
    else
    {
        out << "Translation [ X X X ]" << std::endl;

        out << "Rotation [ X X X X ]" << std::endl;
    }

    out << "]" << std::endl; 
    out << "]" << std::endl; 
}

/* Don't really know what this is, I don't think I changed it significantly */
void ompl::base::ManipulatorStateSpace::registerProjections(void)
{
    class ManipulatorDefaultProjection : public ProjectionEvaluator
    {
    public:

        ManipulatorDefaultProjection(const StateSpace *space) : ProjectionEvaluator(space)
        {
        }

        virtual unsigned int getDimension(void) const
        {
            return space_->as<ManipulatorStateSpace>()->manipulatorDimension_;
        }

        virtual void defaultCellSizes(void)
        {
            cellSizes_.resize(space_->as<ManipulatorStateSpace>()->manipulatorDimension_);
            bounds_ = space_->as<ManipulatorStateSpace>()->getBounds();
            cellSizes_[0] = (bounds_.high[0] - bounds_.low[0]) / magic::PROJECTION_DIMENSION_SPLITS;
            cellSizes_[1] = (bounds_.high[1] - bounds_.low[1]) / magic::PROJECTION_DIMENSION_SPLITS;
            cellSizes_[2] = (bounds_.high[2] - bounds_.low[2]) / magic::PROJECTION_DIMENSION_SPLITS;
        }

        virtual void project(const State *state, EuclideanProjection &projection) const
        {
            memcpy(&projection(0), state->as<ManipulatorStateSpace::StateType>()->as<RealVectorStateSpace::StateType>(0)->values, space_->as<ManipulatorStateSpace>()->manipulatorDimension_ * sizeof(double));
        }
    };

    registerDefaultProjection(ProjectionEvaluatorPtr(dynamic_cast<ProjectionEvaluator*>(new ManipulatorDefaultProjection(this))));
}
