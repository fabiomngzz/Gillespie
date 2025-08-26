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