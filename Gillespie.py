import numpy as np
from scipy.interpolate import PchipInterpolator
from helpers import evalContextVar

# Function for performing a step of a Gillespie simulation for stochastic processes
# Inputs:
#   - context: object of the form {...key: value,...}
#       it stores the parameter values and the variables of the model
#   - reactions: reactions: object of the form 
#       { 
#         'description': str (simple description of the process; not functional), 
#         'probFunc': function,
#         'probFuncVars': list of strings corresponding to the names of the arguments of the probability function; notice that these are found in the context!
#         'oldState': list of strings specifying the starting state(s) of the process; needed to get the subset of nodes that can undergo the update
#         'oldState': list of strings specifying the final state(s) of the process
#       }
# 
# Algorithm: 
#   1. Get the vector storing the state of each node of the system
#   2. Evaluate the rates of the processes; keep only the non-0 ones, i.e. those corresponding to processes that can actually happen 
#        Note: must be done at each step because, in general, they depend on the state of the system
#   3. Check: if no rate is positive, the system reached an absorbing state; in that case, the simulation is stopped
#   4. Perform the Gillespie update as customary:
#        4.1. Normalize the positive rates to 1
#        4.2. Launch a random number, selecting a process
#        4.3. Get the 'oldState' from the reactions object; get all sites in the oldState state; these are the candidates for the update
#        4.4. If there are any candidates: select one at random and update it to 'newState'; update the vector of states of the nodes in the context as well
#        4.5. If there are any candidates, i.e. if the process occurs: update the time randomly according to the rate of the process
#      Note: steps 4.2 to 4.5 are repeated until a valid update is selected
#
# Output: 
#   - abs: bool (the current state is absorbing or not)
#   - object of the form
#       { 
#         't': float
#         'vec': 1D-array-like (vector storing the states of the nodes)
#     }
def GillespieStep(context,reactions):
    vec = evalContextVar(['varVec'],context)[0]
    # Choose a process according to their propensities
    reactsRates = [r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions]
    #print('\t\t',reactsRates)
    # Remove the null rates
    positiveReactsRates = [r for r in reactsRates if r > 0]
    # Check for absorbing state
    absState = False
    if len(positiveReactsRates)==0:
        absState = True
        #print(absState)
        # If the absorbing state is reached, exit
        return absState, {
            't' : 0,
            'vec' : vec
        }

    ratesCum = np.cumsum(reactsRates)
    probsCum = ratesCum / ratesCum[-1]
    #print('\t\t',probsCum)

    while True:
        temp = np.random.rand()
        for ii in range(len(reactions)):
            if temp < probsCum[ii]:
                # Get the chosen reaction
                reactionCurrent = reactions[ii]
                #print('\t\t\t',reactionCurrent['description'])
                oldState = np.array(evalContextVar(reactionCurrent['oldState'],context))
                newState = np.array(evalContextVar(reactionCurrent['newState'],context))
                #print('\t\t\t',oldState)
                # Check if there are candidates for the update; if so, perform the update
                potentialIdxs = [np.where(vec == s)[0] for s in oldState]
                #print('\t\t\tCandidates for update: ',len(potentialIdxs))
                nPotentialUpdates = np.array([len(pI) for pI in potentialIdxs])
                #print('\t\t\t',nPotentialUpdates)
                if np.all(nPotentialUpdates > 0):
                    #print('\t\t\tRandom number ',temp)
                    #print('\t\t\t',reactionCurrent['description'])
                    
                    vecNew = vec.copy()
                    for n, pI in zip(newState,potentialIdxs):
                        idxCurrent = np.random.choice(pI)
                        vecNew[idxCurrent] = n

                    # Update variables; also in context   
                    vec = vecNew
                    context['varVec'] = vec      

                    # Update time
                    tStep = np.log(1/temp)/(ratesCum[-1])
                    return absState, {
                        't' : tStep,
                        'vec' : vec
                    }
                break

# Function that produces a time grid common to all series produced with many repetitions of the Gillespie algorithm
# 
# Motivation: the Gillespie algorithm for the simulation of stochastic processes produces unevenly-sampled
#   data over the time axis. In order to calculate averages, then, it is necessary to interpolate them 
#   on a common time grid, which is produced by the following function.
# 
# Algorithm: 
#   1. The code checks whether each time series is absorbing, which is specified in the input
#   2. The maximum time of the common time grid (tMaxGill) is chose as the minimum maximum time of every non-absorbing time series
#   3. The step of the grid is evaluated given the rates of the microscopic stochastic processes as the inverse of the sum of such rates (times an optional multiplicative factor) (timeScaleTyp)
#   4. The common time grid is produced as an evenly-spaced 1D-array from 0 to tMaxGill with step timeScaleTyp
# 
# Inputs:
#   - seriesObj: object of the form 
#     [...
#       {
#         timeLabel: 1D-array-like
#         ...
#       },
#     ...]
#     it stores the results of each repetition of the dynamics 
#   - context: object of the form {...key: value,...}
#       it stores the parameter values of the model
#   - reactions: object of the form 
#       { 
#         ..., 
#         'probFunc': function,
#         'probFuncVars': list of strings corresponding to the names of the arguments of the probability function
#         ... 
#       }
#   - timeLabel (optional): string
#   - absorningLabel (optional): string
#   - mulF (optional): float
def tVecCommon(seriesObj,context,reactions,timeLabel='time',absorbingLabel='absorbing',mulF=1):
    tVecNonAbs = [obj[timeLabel] for obj in seriesObj if not obj[absorbingLabel]]
    if len(tVecNonAbs)>0:
        tMaxGill = min(max(t) for t in tVecNonAbs)
    else:
        tMaxGill = max(obj[timeLabel][-1] for obj in seriesObj)
    ratesTyp = np.sum([r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions])
    timeScaleTyp = mulF*ratesTyp**-1
    return np.arange(0,tMaxGill,timeScaleTyp)

# Function to interpolate the time series pertaining to different realizations of Gillespie simulations 
# of a stochastic process over a common time grid
# 
# Inputs:
#   - tVecAvg: common time grid
#   - seriesObj: obect of the form
#       [...,
#       {
#         timeLabel: 1D-array-like
#         seriesLabel: 1D-array-like
#       },
#     ...]
#     it stores the results of each repetition of the dynamics
#   - dataLabel: string
#       name of the variable storing the time series
#   - timeLabel (optional): string
#   - absorbingLabel (optional): string
def GillTimeSeriesInterp(tVecAvg,seriesObj,dataLabel,timeLabel='time',absorbingLabel='absorbing'):
    dataInterpVec = []
    for obj in seriesObj:
        dataInterpObj = PchipInterpolator(obj[timeLabel],obj[dataLabel],extrapolate=False)
        dataInterp = dataInterpObj(tVecAvg)
        if obj[absorbingLabel]:
            tMax = obj[timeLabel][-1]
            finalState = obj[dataLabel][-1]
            dataInterp[tVecAvg > tMax] = finalState
        dataInterpVec.append(dataInterp)
    return dataInterpVec