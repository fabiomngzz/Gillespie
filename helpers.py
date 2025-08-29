import numpy as np

def setVec(vec,val):
    vec[:]=val
    return vec

def checkNodeState(vec,idx,vals,labels):
    val = vec[idx]
    labelIdx = np.where( vals == val)[0][0]
    return labels[labelIdx]

def getSubvec(vec,targetState):
    subVec = vec[vec == targetState]
    return subVec

def countSpecies(vec,targetState):
    return len(getSubvec(vec,targetState))

def speciesFrac(vec,targetState):
    return len(getSubvec(vec,targetState))/len(vec)

def evalContextVar(names, context):
    return [eval(name, {}, context) for name in names]

def stationaryMean(x,y,cut):
        x = np.array(x)
        y = np.array(y)
        idxList = np.where(x > cut)[0]
        if len(idxList)==0:
            raise ValueError('No data satisfying the cut-off detected!')
        return np.mean(y[idxList])