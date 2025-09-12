import numpy as np

# Function to quickly set the element of a 1D-array-like object
def setVec(vec,val):
    vec[:]=val
    return vec

# Functionthat returns the quaality of a site of a 1D array given a set of value-label pairs for every possible state of the node
def checkNodeState(vec,idx,vals,labels):
    val = vec[idx]
    labelIdx = np.where( vals == val)[0][0]
    return labels[labelIdx]

# Function to get only the elements of an array that store a given value
def getSubvec(vec,targetState):
    subVec = vec[vec == targetState]
    return subVec

# Function to get the number of elements of a 1D array storing a given value
def countSpecies(vec,targetState):
    return len(getSubvec(vec,targetState))

# Function to get the fraction of elements of a 1D array storing a given value over the total length of the array
def speciesFrac(vec,targetState):
    return len(getSubvec(vec,targetState))/len(vec)

# Function to evaluate the variable(s) in "names" given "context", an object which stores all parameters of the system.
# Inputs:
#   names: list of strings equal to the keys of the "context" object
#   context: object of the form {...key: value,...}
#     it stores the parameter values of the model
def evalContextVar(names, context):
    return [eval(name, {}, context) for name in names]

# Function to evaluate the average of time series y cutting the transient, i.e. for x greater than a given cutoff
def stationaryMean(x,y,cut):
        x = np.array(x)
        y = np.array(y)
        idxList = np.where(x > cut)[0]
        if len(idxList)==0:
            raise ValueError('No data satisfying the cut-off detected!')
        return np.mean(y[idxList])