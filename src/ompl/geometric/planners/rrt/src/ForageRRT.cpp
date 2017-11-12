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

#include "ompl/geometric/planners/rrt/ForageRRT.h"
#include <moveit/ompl_interface/parameterization/manipulator_space/manipulator_model_state_space.h>

/** \brief Try to advance one stepSize towards tryState */
ompl::geometric::ForageRRT::RRT::StepResult ompl::geometric::ForageRRT::RRT::tryStep( ompl::base::State *tryState, 
										      Motion *returnMotion )
{
    Motion *tryToMotion   = new Motion(si_);
    tryToMotion->state = tryState;

    /* Find closest state in the tree */
    Motion *nearestMotion = nn_->nearest(tryToMotion);
    
    /* Find state to add */
    double dist = si_->distance(nearestMotion->state, tryState);
    //std::cout<<"maxDistance_:"<<maxDistance_<<'\n';
    if (dist > maxDistance_)
    {
	/* Cannot connect tryState directly to existing tree, interpolate from nearest neighbor */
        ompl::base::State *interpedState = si_->allocState();
	si_->getStateSpace()->interpolate(nearestMotion->state, tryState, maxDistance_ / dist, interpedState);

        if (si_->checkMotion(nearestMotion->state, interpedState))
        {
	    /* New new state is valid (can connect without collision to existing state) */
	    addNode(interpedState, nearestMotion);
            returnMotion->state = interpedState;
	    returnMotion->parent = nearestMotion;
	    return STEP_PROGRESS;	
        }
	else
	{
	    return STEP_COLLISION;
	}
	    
    }

    /* Trystate and state to connect to it from are close enough to connect */
    ompl::base::State *interpedState = si_->allocState();
    si_->getStateSpace()->interpolate(nearestMotion->state, tryState, maxDistance_ / dist, interpedState);

    addNode(interpedState, nearestMotion);
    returnMotion->state = interpedState;                 
    returnMotion->parent = nearestMotion;

    return STEP_REACHED;
}

/** \brief Try to advance one stepSize towards goalState. Take step from top of Goal Map */
ompl::geometric::ForageRRT::RRT::StepResult ompl::geometric::ForageRRT::RRT::tryStepToGoal( ompl::base::State *goalState, 
									                    Motion *returnMotion )
{
    Motion *goalMotion   = new Motion(si_);
    goalMotion->state = goalState;

    /* find closest state in the tree to goal or random */
    Motion* fromMotion;

    if(rng_.uniform01() < .65 /*Magic number - make configurable */)
    {
    	fromMotion = getBestGoalMotion();
    }
    else
    {
        fromMotion = getRandomGoalMotion();
    }

    /* find state to add */
    double dist = si_->distance(fromMotion->state, goalState);

    if (dist > maxDistance_)
    {
        /* Distance between the two points is not within fineStepSize_, so goal cannot be reached this step */
	/* Interpolate between the two states */
        ompl::base::State *interpedState = si_->allocState();
        si_->getStateSpace()->interpolate(fromMotion->state, goalState, maxDistance_ / dist, interpedState);

        if (si_->checkMotion(fromMotion->state, interpedState))
        {
	    /* New new state is valid (can connect without collision to existing state) */
            addNode(interpedState, fromMotion);
	    returnMotion->state = interpedState;
	    returnMotion->parent = fromMotion;
            return STEP_PROGRESS;
        }
        else
        {
            return STEP_COLLISION;
        }
            
    }

    /* Goal and state to connect to it from are close enough to connect */
    ompl::base::State *interpedState = si_->allocState();
    si_->getStateSpace()->interpolate(fromMotion->state, goalState, maxDistance_ / dist, interpedState);

    addNode(interpedState, fromMotion);
    returnMotion->state = interpedState;
    returnMotion->parent = fromMotion;
    return STEP_REACHED;
}

/** \brief Pull best node from goal heap not yet attempted to go to goal from */
ompl::geometric::ForageRRT::RRT::Motion* ompl::geometric::ForageRRT::RRT::getBestGoalMotion( void )
{
    /* Pull the top of the goal heap */
    std::map<float, Motion*>::iterator it = GoalHeap_.begin();
    Motion *bestMotion = it->second;
    /* Keep pulling until we have gotten one that hasn't been tried. Remove all others */
    while(bestMotion->triedToGoal && GoalHeap_.size()>1){
            GoalHeap_.erase(bestMotion->value);
            it = GoalHeap_.begin();
            bestMotion = it->second;
    }
    bestMotion->triedToGoal = true;
    return bestMotion;
}

/** \brief Pull random node from goal heap not yet attempted to go to goal from */
ompl::geometric::ForageRRT::RRT::Motion* ompl::geometric::ForageRRT::RRT::getRandomGoalMotion( void )
{
    /* Pull the top node of the goal heap */
    std::map<float, Motion*>::iterator it = GoalHeap_.begin();
    std::advance( it, (rand() % GoalHeap_.size()) );
    Motion *bestMotion = it->second;
    /* Keep pulling until we have gotten one that hasn't been tried to goal. Remove all others */
    while(bestMotion->triedToGoal && GoalHeap_.size()>1){
            GoalHeap_.erase(bestMotion->value);
            it = GoalHeap_.begin();
	    std::advance( it, (rand() % GoalHeap_.size()) );
            bestMotion = it->second;
    }
    bestMotion->triedToGoal = true;
    return bestMotion;
}

/** \brief Add node to the RRT, filling out its fields and adding it to the NN and GoalHeap structures */
void ompl::geometric::ForageRRT::RRT::addNode( ompl::base::State* newState, Motion* parentMotion )
{
    /* create a motion */
    Motion *newMotion = new Motion(si_);
    si_->copyState(newMotion->state, newState);
    newMotion->parent = parentMotion;
    
    /* cost of a new node is its distance since the start */
    if (parentMotion != NULL)
        newMotion->cost = parentMotion->cost + si_->distance(newState, parentMotion->state);
    else
        newMotion->cost = 0;

    /* Fill out the motion value (distance from goal) and that it has not been tried to goal */
    newMotion->value = motionValue( newMotion );
    newMotion->triedToGoal = false;

    /* Add to goal heap and nearest neighbors tree */
    GoalHeap_.insert(NodeValuePair(newMotion->value, newMotion));
    nn_->add(newMotion);
}


/** \brief Return the value of a given motion, its distance to goal */
double ompl::geometric::ForageRRT::RRT::motionValue( const Motion* motion )
{
    assert(goal_);
    return si_->distance(motion->state, goal_);
}

/** \brief Follow parents of solution node to root and add to given path */
void ompl::geometric::ForageRRT::RRT::tracePath(Motion *solution, PathGeometric *path)
{
    /* construct the solution path */
    std::vector<Motion*> motionPath;
    while (solution != NULL)
    {
        motionPath.push_back(solution);
        solution = solution->parent;
    }

    /* set the solution path */
    for (int i = motionPath.size() - 1 ; i >= 0 ; --i)
        path->append(motionPath[i]->state);
}

/** \brief Constructor. Initializes relevant parameters. Setup also needs to be called*/
ompl::geometric::ForageRRT::ForageRRT(const base::SpaceInformationPtr &si) : base::Planner(si, "ForageRRT")
{
    specs_.approximateSolutions = true;
    specs_.directed = true;

    fineRRTGoalBias_ = 0.05;
    lastGoalMotion_ = NULL;
    fineStepSize_ = 0.01;
    coarseStepSize_ = .2;
    coarseRRTInitialSize_ = 100;//30;
    coarseRRTGoalBias_ = .10;
    maxFineNumCollisions_ = 20;
    maxNumFineFailures_ = 10;
    coarseRRTIncreaseSize_ = coarseRRTInitialSize_/3;
}

/** \brief Deconstructor. Frees memory occupied by planner */
ompl::geometric::ForageRRT::~ForageRRT(void)
{
    freeMemory();
}

/** \brief Resets the planner including freeing memory */
void ompl::geometric::ForageRRT::clear(void)
{
    Planner::clear();
    sampler_.reset();
    coarseRRT_->clear();
    fineRRT_->clear();
    lastGoalMotion_ = NULL;
}

/** \brief Setup Forage RRT - create Coarse and Fine RRT and then set each one up */
void ompl::geometric::ForageRRT::setup(void)
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(fineStepSize_);

    coarseRRT_ = new RRT(si_, coarseStepSize_);
    fineRRT_ = new RRT(si_, fineStepSize_);

    coarseRRT_->setup();
   // std::cout<<coarseRRT_->getSize()<<'\n';
    fineRRT_->setup();
   // std::cout<<fineRRT_->getSize()<<'\n';
}

/** \brief Routine to free memory used by RRTs */
void ompl::geometric::ForageRRT::freeMemory(void)
{
    coarseRRT_->freeMemory();
    fineRRT_->freeMemory();
}

/** \brief Grow the input RRT by size using goalBias (used for Coarse RRT)*/
void ompl::geometric::ForageRRT::growRRT(RRT *rrt, unsigned int size, double goalBias)
{
    base::Goal *goal = pdef_->getGoal().get();
    base::GoalSampleableRegion *goal_s = dynamic_cast<base::GoalSampleableRegion*>(goal);
    ompl::base::State *toState = si_->allocState();
    RRT::Motion *newMotion = new RRT::Motion();
    double initialSize = rrt->getSize();

    /* Repeat until desired size increase is achieved */
    while ((rrt->getSize() - initialSize) < size )
    {
        if (goal_s && rng_.uniform01() < goalBias && goal_s->canSample())
        {
	    /* Here we try to move toward the goal */
            goal_s->sampleGoal(toState);
	        rrt->tryStepToGoal(toState, newMotion);
        }
        else
        {
	    /* Here we try to move toward a random point */
            sampler_->sampleUniform(toState);
            rrt->tryStep(toState, newMotion);
        }
    }
}

/** \brief Solve routine for the Forage RRT, main routine for this planner */
ompl::base::PlannerStatus ompl::geometric::ForageRRT::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    
    std::cout<<"retrieve goal state"<<std::endl;
    /* Retrieve goal state */
    base::Goal *goal = pdef_->getGoal().get();
    base::GoalSampleableRegion *goal_s = dynamic_cast<base::GoalSampleableRegion*>(goal);
    const ompl::base::State *goalState = pis_.nextGoal(ptc);  
 
    std::cout<<"set goal state for coarseRRT"<<std::endl;
    coarseRRT_->setGoal(goalState);
    std::cout<<"set goal state for fineRRT"<<std::endl;
    fineRRT_->setGoal(goalState);

    /* Add all start nodes to coarse RRT */
    std::cout<<"add start nodes to coarseRRT"<<std::endl;
    while (const base::State *st = pis_.nextStart())
    {
        coarseRRT_->addNode(const_cast<base::State* const>(st), NULL);
    }
    
    //std::cout<<coarseRRT_->
    /* Check that we have at least one start state */
    if (coarseRRT_->getSize() == 0)
    {
        OMPL_ERROR("There are no valid initial states!");
        return base::PlannerStatus::INVALID_START;
    }

    /* Allocate state sampler if we do not have one */
    std::cout<<"allocate state sampler"<<std::endl;
    if (!sampler_)
        sampler_ = si_->allocStateSampler();

    OMPL_INFORM("Starting with %u states already in datastructure", coarseRRT_->getSize());

    /* Allocate and initialize other planner variables */
    std::cout<<"intialize other planner variables"<<std::endl;
    RRT::Motion *fineSolution  = NULL;
    RRT::Motion *coarseSolution = NULL;
    RRT::Motion *approxFineSol = NULL;
    RRT::Motion *approxCoarseSol = NULL;

    double  approxdif = std::numeric_limits<double>::infinity();

    base::State *toState = si_->allocState();
    RRT::Motion *newMotion = new RRT::Motion();
    RRT::Motion *fineSeed = new RRT::Motion();

    unsigned int numFineFailures = 0;
    unsigned int fineNumCollisions = 0;

    /* Grow the coarse RRT to its initial size */
    std::cout<<"grow the coarseRRT to its initial size"<<std::endl;
    growRRT(coarseRRT_, coarseRRTInitialSize_, coarseRRTGoalBias_);

    /* Seed the fine RRT with the best node towards goal from the coarse RRT */	
    std::cout<<"Seed the fine RRT with the best node towards goal from the coarse RRT"<<std::endl;
    fineSeed = coarseRRT_->getBestGoalMotion();
    fineRRT_->addNode(fineSeed->state, NULL);

    /* Repeat until the goal state has been found */
    while (ptc == false)
    {
	    RRT::StepResult result;
        /* Sample random state (with goal biasing) for fine RRT */
        if (goal_s && rng_.uniform01() < fineRRTGoalBias_ && goal_s->canSample())
	    {
 	        /* Sample a goal and try to reach it from the existing tree */
	        goal_s->sampleGoal(toState);
	        result = fineRRT_->tryStepToGoal( toState, newMotion );
	    }
        else
	    {
            /* Sample a random state and try to reach it from the existing tree */
            sampler_->sampleUniform(toState);
	        result = fineRRT_->tryStep( toState, newMotion );
        }

        if ( result == RRT::STEP_REACHED || result == RRT::STEP_PROGRESS )
        {
            /* We've at least taken a step toward our sampled state. Let's see if the newly added node to the Fine RRT is the goal*/
            double dist = 0.0;
            if (goal->isSatisfied(newMotion->state, &dist))
            {
                /* New node is goal! The RRT can now solve the problem */
                approxdif = dist;
                fineSolution = newMotion;
		        coarseSolution = fineSeed;
                break;
            }
            if (dist < approxdif)
            {
		/* Not quite at goal, but new node is closer than any before, we have a new 'best' plan */
                approxdif = dist;
                approxFineSol = newMotion;
		        approxCoarseSol = fineSeed;
            }
        }

	    else
	    {
	        /* Attempted step has produced a collision */
            /* Increment number of collisions encountered by the fine RRT */
	        fineNumCollisions++;
	        if (fineNumCollisions == maxFineNumCollisions_)
	        {
		        /* Collision limit for Fine RRT reached, let's give up and try another fine RRT from a different direction */
		        fineNumCollisions = 0;
		        numFineFailures++;
		        if (numFineFailures == maxNumFineFailures_)
		        {
		            /* Too many fine RRTs have failed, let's grow coarse RRT to replenish well of seed nodes for Fine RRTs */
		            numFineFailures = 0;
		            growRRT(coarseRRT_, coarseRRTIncreaseSize_, coarseRRTGoalBias_);
		        }
   	            /* Seed the new fine RRT with the best node towards goal from the coarse RRT */
                fineSeed = coarseRRT_->getBestGoalMotion();
		        fineRRT_->clear();
                fineRRT_->addNode(fineSeed->state, NULL);
	        }
	    }
    }

    /* No goal state was found, plan not solved */
    bool solved = false;
    bool approximate = false;

    if (fineSolution == NULL)
    {
	/* Let's at least output best solution */
        std::cout<<"Let's at least output best solution"<<std::endl;
        fineSolution = approxFineSol;
	    coarseSolution = approxCoarseSol;
        approximate = true;
    }

    if (fineSolution != NULL)
    {
	/* Return solution path, be it 'best without success' or successful */
        std::cout<<"Return solution path"<<std::endl;
        lastGoalMotion_ = fineSolution;
        PathGeometric *path = new PathGeometric(si_); 
	    coarseRRT_->tracePath(coarseSolution, path);
	    fineRRT_->tracePath(fineSolution, path);
        pdef_->addSolutionPath(base::PathPtr(path), approximate, approxdif);
        solved = true;
        //std::vector<base::State *> &states = path->getStates();
        //const base::SpaceInformationPtr &si = path->getSpaceInformation();
        //si->getStateSpace()->as<ompl_interface::ManipulatorModelStateSpace>()->manipulator_state_space_->printState(states.back()->as<ompl_interface::ManipulatorModelStateSpace::StateType>()->manipulator_state_,std::cout);
    }
    
    std::cout<<"exit ForageRRT solve"<<std::endl;
    return base::PlannerStatus(solved, approximate);
}

/** \brief In addition to regular planner data, add coarse RRT nodes */
void ompl::geometric::ForageRRT::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<RRT::Motion*> motions;
    if (coarseRRT_->nn_)
        coarseRRT_->nn_->list(motions);

    if (lastGoalMotion_)
        data.addGoalVertex(base::PlannerDataVertex(lastGoalMotion_->state));

    for (unsigned int i = 0 ; i < motions.size() ; ++i)
    {
        if (motions[i]->parent == NULL)
            data.addStartVertex(base::PlannerDataVertex(motions[i]->state));
        else
            data.addEdge(base::PlannerDataVertex(motions[i]->parent->state),
                         base::PlannerDataVertex(motions[i]->state));
    }
}
