import numpy as np
from scipy.interpolate import PchipInterpolator
from helpers import evalContextVar, speciesFrac
from CPRSust_metapop_rates import consumersEncounterFactor

def GillespieStep(context,reactions):
    vec = evalContextVar(['varVec'],context)[0]
    # Choose a process according to their propensities
    reactsRates = [r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions]
    # Remove the null rates
    positiveReactsRates = [r for r in reactsRates if r > 0]
    #print('\t\t',reactsRates)
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

def tVecCommon(seriesObj,context,reactions,timeLabel='time',absorbingLabel='absorbing',mulF=1):
    tVecNonAbs = [obj[timeLabel] for obj in seriesObj if not obj[absorbingLabel]]
    if len(tVecNonAbs)>0:
        tMaxGill = min(max(t) for t in tVecNonAbs)
    else:
        tMaxGill = max(obj[timeLabel][-1] for obj in seriesObj)
    ratesTyp = np.sum([r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions])
    timeScaleTyp = mulF*ratesTyp**-1
    return np.arange(0,tMaxGill,timeScaleTyp)

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