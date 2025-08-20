import numpy as np
from scipy.interpolate import PchipInterpolator
from helpers import evalContextVar

def GillespieStep(context,reactions):
    vec = evalContextVar(['varVec'],context)[0]
    # Choose a process according to their propensities
    reactsRates = [r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions]
    #print('\t\t',reactsRates)
    ratesCum = np.cumsum(reactsRates)
    probsCum = ratesCum / ratesCum[-1]
    #print('\t\t',ratesCum)
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
                # Check if there are neighbors of the required state
                requiredNeighborState = evalContextVar(reactionCurrent['neighborState'],context)
                # print('\t\t\t',requiredNeighborState)
                if requiredNeighborState:
                    checkNeighborState = np.any([np.where(vec == s)[0] for s in requiredNeighborState])
                else:
                    checkNeighborState = True
                #print('\t\t\t',checkNeighborState)
                nPotentialUpdates = np.array([len(pI) for pI in potentialIdxs])
                #print('\t\t\t',nPotentialUpdates)
                if np.all(nPotentialUpdates > 0) and checkNeighborState:
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
                    return {
                        't' : tStep,
                        'vec' : vec
                    }
                break

def tVecCommon(tVec,context,reactions,mulF=1):
    tMaxGill = min(max(t) for t in tVec)
    ratesTyp = np.sum([r['probFunc'](*evalContextVar(r['probFuncVars'],context)) for r in reactions])
    timeScaleTyp = mulF*ratesTyp**-1
    return np.arange(0,tMaxGill,timeScaleTyp)

def GillTimeSeriesInterp(tAvg,tVec,dataVec):
    dataInterpVec = []
    for t, data in zip(tVec,dataVec):
        dataInterp = PchipInterpolator(t,data)
        dataInterpVec.append(dataInterp(tAvg))
    return dataInterpVec