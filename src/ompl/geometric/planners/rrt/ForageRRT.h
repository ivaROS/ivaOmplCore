/*********************************************************************
* Software License Agreement (BSD License)
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
*   * Neither the name of the Willow Garage nor the names of its
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

#ifndef OMPL_GEOMETRIC_PLANNERS_RRT_FORAGERRT_
#define OMPL_GEOMETRIC_PLANNERS_RRT_FORAGERRT_

#include "ompl/geometric/planners/PlannerIncludes.h"
#include "ompl/datastructures/NearestNeighbors.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include <limits>

namespace ompl
{
    namespace geometric
    {
        /**
           @anchor ForageRRT
           @par Short description
           The ForageRRT expands on the RRT by using two separate but connected trees in the search. One tree is a large step-size RRT which is sparse but covers a wide area of the search space
           while the other is a fine step-size RRT which starts from a node in the large step-size RRT and tries to find its way to goal. Once a fine step-size RRT finds the goal, the path is
           traced backward through the fine step-size RRT to its large step-size RRT parent node and then all the way back to the start state.
           @par External documentation
           Keselman, Leo, Erik Verriest, and Patricio Vela. "Forage RRT—An efficient approach to task-space goal planning for high dimensional systems." Robotics and Automation (ICRA), 2014 IEEE International Conference on. IEEE, 2014.
        */

        /** \brief Rapidly-exploring Random Trees */
        class ForageRRT : public base::Planner
        {
        public:
	    /** \brief A state for keeping track of a single RRT's information. Useful here because we will be combining multiple RRTs */
	    class RRT //: public base::Planner
	    {
    	    public:
		/** \brief Representation of a motion
                This only contains pointers to parent motions as we
                only need to go backwards in the tree. */
                class Motion
                {
                public:
                    
		    Motion(void) : state(NULL), parent(NULL)
                    {
                    }

                    /** \brief Constructor that allocates memory for the state */
                   Motion(const base::SpaceInformationPtr &si) : state(si->allocState()), parent(NULL)
                   {
                   }

		    /** \brief Destructor */
                    ~Motion(void)
                    {
                    }
                    
        //            Motion() = default;

                   /** \brief Constructor that allocates memory for the state */
        //            Motion(const base::SpaceInformationPtr &si) : state(si->allocState())
        //            {
        //            }

        //            ~Motion() = default;
		    
                    /** \brief The state contained by the motion */
                    base::State       *state;

                    /** \brief The parent motion in the exploration tree */
                    Motion           *parent;

		    /** \brief The state contained by the motion */
             //       base::State *state{nullptr};

                    /** \brief The parent motion in the exploration tree */
             //       Motion *parent{nullptr};
		     
                    /** \brief The cost of the state, the distance to get there from the start along its parents */
	 	    double 		cost;

		    /** \brief The value of the state */
		    double             value;

		    /** \brief Says whether motion has been tried as originator to goal( should only be tried once ) */
		    bool         triedToGoal;
                };

		typedef std::map<float, Motion*> GoalValueMap;
                typedef std::pair<float, Motion*> NodeValuePair;
 
	        typedef enum {
	            STEP_COLLISION,  /**< Collided with obstacle. Not added */
      		    STEP_REACHED,    /**< The configuration that we grow to is less than stepSize away from node we grow from. */
      		    STEP_PROGRESS    /**< One node added */
   		}  StepResult;

		/** \brief RRT constructor */
		RRT(const base::SpaceInformationPtr &si, double stepSize)
		{
		    si_ = si;
		    maxDistance_ = stepSize;
		    goal_ = NULL;
		}

		/** \brief RRT destructor */
                virtual ~RRT(void)
		{
		    freeMemory();
		}

		/** \brief Set goal node class variable for RRT */
		void setGoal(const ompl::base::State *goal)
		{
		    goal_ = si_->allocState();
		    si_->copyState(goal_, goal);
		}

		/** \brief Setup Forage RRT - create Coarse and Fine RRT and then set each one up */
		void setup(void)
		{
		    if (!nn_)
                        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors_leo<Motion *>(si_->getStateSpace()));//this
		    nn_->setDistanceFunction(boost::bind(&RRT::distanceFunction, this, /*std::placeholders::*/_1, /*std::placeholders::*/_2));
                    //nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                    //         {
                    //            return distanceFunction(a, b);
                    //        });
		}

		/** \brief Clear the RRT object. All settings stay the same, including goal */
		void clear(void)
		{
    		    freeMemory();
    		    if (nn_)
        	        nn_->clear();
		}

		/** \brief Compute distance between motions (actually distance between contained states) */
                double distanceFunction(const Motion* a, const Motion* b) const
                {
                    return si_->distance(a->state, b->state);
                }

		/** \brief Try to advance one stepSize towards tryState */
		StepResult tryStep( ompl::base::State *tryState, Motion *returnMotion );

                /** \brief Try to advance one stepSize towards goalState */
                StepResult tryStepToGoal( ompl::base::State *goalState, Motion *returnMotion );

		/** \brief Return size of RRT as number of nodes */
		unsigned int getSize( void )
		{
		    return nn_->size();
		}

		/** \brief Pull best node from goal heap not yet attempted to go to goal from */
		Motion* getBestGoalMotion( void );

		/** \brief Pull random node from goal heap not yet attempted to go to goal from */
                Motion* getRandomGoalMotion( void );

		/** \brief Add node to the RRT, filling out its fields and adding it to the NN and GoalHeap structures */
		void addNode( ompl::base::State* newState, Motion* parentMotion );

		/** \brief Follow parents of solution node to root and add to given path */
		void tracePath( Motion *solution, PathGeometric *path );
    		     
		/** \brief A nearest-neighbors datastructure containing the tree of motions */
            	boost::shared_ptr< NearestNeighbors<Motion*> > nn_;
		/** \brief Set a different nearest neighbors datastructure */
                template<template<typename T> class NN>
                void setNearestNeighbors(void)
                {
                    nn_.reset(new NN<Motion*>());
                }
	
		/** \brief Free RRT Memory used */
		void freeMemory(void)
		{
	 	    if (nn_)
                    {
                        std::vector<Motion*> motions;
                        nn_->list(motions);
                        for (unsigned int i = 0 ; i < motions.size() ; ++i)
                        {
                            if (motions[i]->state)
			    {
                                si_->freeState(motions[i]->state);
			    }
                            delete motions[i];
                        }
                    }
		    GoalHeap_.clear();
		}
		
	    protected:	

		//Protected member variables
		/** \brief Step size for RRT */
		double range_;
		double maxDistance_;
		base::SpaceInformationPtr  si_;
		ompl::base::State* goal_;
		GoalValueMap GoalHeap_;

		/** \brief Return the value of a given motion */
		virtual double motionValue( const Motion* motion );
	        /** \brief The random number generator */
            	RNG                                            rng_;
	

            };

            /** \brief Constructor */
            ForageRRT(const base::SpaceInformationPtr &si);

	    /** \brief Destructor */
            virtual ~ForageRRT(void);

	    /** \brief In addition to regular planner data, add coarse RRT nodes */
            virtual void getPlannerData(base::PlannerData &data) const;

	    /** \brief Grow the input RRT by size using goalBias (used for Coarse RRT)*/
	    void growRRT(RRT *rrt, unsigned int size, double goalBias);

	    /** \brief Solve routine for the Forage RRT, main routine for this planner */
            virtual base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc);

	    /** \brief Resets the planner including freeing memory */
            virtual void clear(void);

            /** \brief Set the fine RRT goal bias*/
            void setFineGoalBias(double goalBias)
            {
                fineRRTGoalBias_ = goalBias;
            }

            /** \brief Get the fine RRT goal bias the planner is using */
            double getFineGoalBias(void) const
            {
                return fineRRTGoalBias_;
            }

	    /** \brief Set the fine RRT goal bias*/
            void setCoarseGoalBias(double goalBias)
            {
                coarseRRTGoalBias_ = goalBias;
            }

            /** \brief Get the fine RRT goal bias the planner is using */
            double getCoarseGoalBias(void) const
            {
                return coarseRRTGoalBias_;
            }

	    /** \brief Set the initial size for the coarse RRT*/
            void setCoarseInitialSize(unsigned int initSize)
            {
                coarseRRTInitialSize_ = initSize;
            }

	    /** \brief Get the initial size for the coarse RRT*/
            unsigned int getCoarseInitialSize(unsigned int initSize) const
            {
                return coarseRRTInitialSize_;
            }

	    /** \brief Set the initial size for the coarse RRT*/
            void setMaxFineCollisions(unsigned int maxCollisions)
            {
                maxFineNumCollisions_ = maxCollisions;
            }

            /** \brief Get the initial size for the coarse RRT*/
            unsigned int getMaxFineCollisions( void ) const
            {
                return maxFineNumCollisions_;
            }

            /** \brief Set the initial size for the coarse RRT*/
            void setMaxFineFailures(unsigned int maxFailures)
            {
                maxNumFineFailures_ = maxFailures;
            }

            /** \brief Get the initial size for the coarse RRT*/
            unsigned int getMaxFineFailures( void ) const
            {
                return maxNumFineFailures_;
            }

	    /** \brief Set the step sizes the planner is supposed to use.

                Coarse should be 20-70 times bigger than fine */
            void setCoarseStepSize(double coarseStepSize)
            {
                coarseStepSize_ = coarseStepSize;
            }

            /** \brief Get the coarse step size the planner is using */
            double getCoarseStepSize(void) const
            {
                return coarseStepSize_;
            }

	    /** \brief Set the fine step size the planner is using */
	    void setFineStepSize(double fineStepSize)
            {   
                fineStepSize_ = fineStepSize;
            }
            
            /** \brief Get the fine step size the planner is using */
            double getFineStepSize(void) const
            {
                return fineStepSize_;
            }

	    /** \brief Set the increase amount of the coarse RRT after a number of unsuccessful fine RRTs */
	    void setCoarseIncreaseSize(double increaseSize)
            {
                coarseRRTIncreaseSize_ = increaseSize;
            }

            /** \brief Get the increase amount of the coarse RRT after a number of unsuccessful fine RRTs */
            double getCoarseIncreaseSize(void) const
            {
                return coarseRRTIncreaseSize_;
            }

            virtual void setup(void);

       protected:

            /** \brief Free the memory allocated by this planner */
            void freeMemory(void);

            /** \brief State sampler */
            base::StateSamplerPtr                          sampler_;
 
            /** \brief The random number generator */
            RNG                                            rng_;

            /** \brief The most recent goal motion.  Used for PlannerData computation */
            RRT::Motion                                    *lastGoalMotion_;

	    /** \brief Coarse step size RRT for fast exploration */
	    RRT						   *coarseRRT_;

	    /** \brief Fine step size RRT for connecting to goal */
            RRT                                            *fineRRT_;

	    /** \brief Step Size for coarse RRT */
	    double 					   coarseStepSize_;

            /** \brief Step Size for fine RRT */
            double                                         fineStepSize_;

	    /** \brief Goal bias for the fine RRT */
	    double                                         fineRRTGoalBias_;
    
	    /** \brief Initial size for the coarse RRT */
            unsigned int                                   coarseRRTInitialSize_;
    
            /** \brief Goal bias for the coarse RRT */
	    double                                         coarseRRTGoalBias_;
    
	    /** \brief Number of fine RRT collisions to tolerate before giving up and moving on */
	    unsigned int				   maxFineNumCollisions_;
    
	    /** \brief Number of fine RRT give-ups to tolerate before growing coarse tree some more */
	    unsigned int				   maxNumFineFailures_;

	    /** \brief Number of nodes to grow coarse tree by in case maxNumFineFailures_ is reached */
	    unsigned int 				   coarseRRTIncreaseSize_;

        };

    }
}

#endif
