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
#include "ompl/tools/config/MagicConstants.h"

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
    const StateType *rstate = static_cast<const StateType*>(state);
    for (unsigned int i = 0 ; i < manipulatorDimension_ ; ++i)
    {
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
    Eigen::Quaterniond pose1_orientation;
    //Eigen::VectorXd pose1_orientation(4);
	Eigen::VectorXd pose2_position(3);
    Eigen::Quaterniond pose2_orientation;
    //Eigen::VectorXd pose2_orientation(4);

	if (!mState1->eefPoseKnown)
	{
        Eigen::VectorXd temp(7);
		temp = getEefPose(mState1);
        pose1_position << temp(0), temp(1), temp(2);
        //pose1_orientation << temp(3), temp(4), temp(5),temp(6);
        pose1_orientation.x() = temp(3);
        pose1_orientation.y() = temp(4);
        pose1_orientation.z() = temp(5);
        pose1_orientation.w() = temp(6);

        std::cout<<"position of from_state(x,y,z):"<<temp(0)<<temp(1)<<temp(2)<<std::endl;
        std::cout<<"orientation of from_state(x,y,z,w):"<<temp(3)<<temp(4)<<temp(5)<<temp(6)<<std::endl;

		const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
		pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ();
        //pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose2->rotation().w;
        pose2_orientation.x() = eefPose2->rotation().x;
        pose2_orientation.y() = eefPose2->rotation().y;
        pose2_orientation.z() = eefPose2->rotation().z;
        pose2_orientation.w() = eefPose2->rotation().w;

        std::cout<<"position of to_state(x,y,z):"<<eefPose2->getX()<<eefPose2->getY()<<eefPose2->getZ()<<std::endl;
        std::cout<<"orientation of to_state(x,y,z,w):"<<eefPose2->rotation().x<<eefPose2->rotation().y<<eefPose2->rotation().z<<eefPose2->rotation().w<<std::endl;
	}
	else if (!mState2->eefPoseKnown)
	{
        Eigen::VectorXd temp(7);
		const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
		pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ(), eefPose1->rotation().x;
        //pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
        pose1_orientation.x() = eefPose1->rotation().x;
        pose1_orientation.y() = eefPose1->rotation().y;
        pose1_orientation.z() = eefPose1->rotation().z;
        pose1_orientation.w() = eefPose1->rotation().w;

		temp = getEefPose(mState2);
        pose2_position << temp(0), temp(1), temp(2);
        //pose2_orientation << temp(3), temp(4), temp(5), temp(6);
        pose2_orientation.x() = temp(3);
        pose2_orientation.y() = temp(4);
        pose2_orientation.z() = temp(5);
        pose2_orientation.w() = temp(6);
	}
	else
	{
		const SE3StateSpace::StateType* eefPose1 = &(mState1->endEffectorPose());
        pose1_position << eefPose1->getX(), eefPose1->getY(), eefPose1->getZ(), eefPose1->rotation().x;
        //pose1_orientation << eefPose1->rotation().x, eefPose1->rotation().y, eefPose1->rotation().z, eefPose1->rotation().w;
        pose1_orientation.x() = eefPose1->rotation().x;
        pose1_orientation.y() = eefPose1->rotation().y;
        pose1_orientation.z() = eefPose1->rotation().z;
        pose1_orientation.w() = eefPose1->rotation().w;

		const SE3StateSpace::StateType* eefPose2 = &(mState2->endEffectorPose());
		pose2_position << eefPose2->getX(), eefPose2->getY(), eefPose2->getZ(), eefPose2->rotation().x;
        //pose2_orientation << eefPose2->rotation().x, eefPose2->rotation().y, eefPose2->rotation().z, eefPose1->rotation().w;
        pose2_orientation.x() = eefPose2->rotation().x;
        pose2_orientation.y() = eefPose2->rotation().y;
        pose2_orientation.z() = eefPose2->rotation().z;
        pose2_orientation.w() = eefPose2->rotation().w;
	}

	dist1 = (pose2_position-pose1_position).norm();

    Eigen::MatrixXd R1 = pose1_orientation.normalized().toRotationMatrix();
    Eigen::MatrixXd R2 = pose2_orientation.normalized().toRotationMatrix();
    Eigen::MatrixXd R = (R1 * R2.transpose()).log();
    for (int i = 0; i < R.rows(); i++)
        for(int j = 0; j < R.cols(); j++)
            dist2 = dist2 + R(i,j) * R(i,j); 
    dist2 = sqrt(dist2);


    //dist2 = 1.0 - fabs(pose1_orientation.transpose() * pose2_orientation);
    //dist2 = std::min((pose2_orientation-pose1_orientation).norm(),(pose2_orientation+pose1_orientation).norm());
    std::cout<<"dist1(position):"<<dist1<<std::endl;
    //std::cout<<"dist2(orientation):"<<dist2<<std::endl;
    //std::cout<<"orientation1"<<pose1_orientation<<std::endl;
    //std::cout<<"orientation2"<<pose2_orientation<<std::endl;
    }
    
    //if(dist2 < 0.3)
        std::cout<<"dist2(orientation):"<<dist2<<std::endl;

    if(dist1 > 0.03)
        return dist1;
    else
        return dist2 / 70;


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
    std::cout<<"enter manipulator interpolate function"<<std::endl;
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
	assert(mStateTo->eefPoseKnown);

        Eigen::VectorXd poseFrom = getEefPose(mStateFrom);
	Eigen::VectorXd poseTo(3);

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

	Eigen::VectorXd properVector = diff.normalized();
	Eigen::VectorXd actualVectorDiff = poseNew-poseFrom;
	Eigen::VectorXd actualVector = actualVectorDiff.normalized();
		//double dotProduct = properVector.dot(actualVector);
		//std::cout<<"<-----Dot product: "<<dotProduct<<" ----->"<<std::endl;
		//std::cin.get();
		//std::cout<<"Assigning to new state"<<std::endl;
	setJoints(mState, jointsInterp);
		//std::cout<<"Done assigning."<<std::endl;
    }
}

/* \brief Return the Jacobian pseudo-inverse of the manipulator instance of the class */
Eigen::MatrixXd ompl::base::ManipulatorStateSpace::getPseudoInvJacobian(const State *inputState) const
{

    const StateType* mInputState = static_cast<const StateType*>(inputState);

    Eigen::MatrixXd Jaclin = manipulatorState_->getJacobian(getEigenJoints(mInputState));
    //std::cout<<"Jacobian"<<std::endl;
    //std::cout<<Jaclin<<std::endl; 
    Eigen::MatrixXd JaclinT = Jaclin.transpose();
    Eigen::MatrixXd Jt;
    Eigen::MatrixXd JJt = (Jaclin*JaclinT);
    //std::cout<<"JJt"<<std::endl;
    //std::cout<<JJt<<std::endl;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(JJt);
    //std::cout<<"lu.inverse"<<std::endl;
    //std::cout<<lu.inverse()<<std::endl;
    Jt = JaclinT*( lu.inverse() );

    return Jt;
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
