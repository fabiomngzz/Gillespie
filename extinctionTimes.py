import numpy as np

# Function to evaluate the times to extinction of a single time series
# 
# Algorithm: the code finds the two time points in between which the y series crosses the threshold 
# and evaluates the extinction time performing a linear estimation of the time series passing through 
# those points
#
# Inputs:
#   t: 1D-array-like
#   y: 1D-array-like
#   tol: tolerance with which to evaluate the proximity of the values to 0
def extinctionTime_single(t,y,tol=1e-3):
    # Get all the indexes for which the resource is 0
    idxExtVec = np.where(np.array(y) <= tol)[0]
    if len(idxExtVec)==0:
        return None
    idxExt = idxExtVec[0]
    # Perform a simple linear interpolation to better estimate the extinction time
    tExt_temp = [t[idxExt-1],t[idxExt]]
    yExt_temp = [y[idxExt-1],y[idxExt]]
    return tExt_temp[0] + yExt_temp[0]*(tExt_temp[1]-tExt_temp[0])/(yExt_temp[0]-yExt_temp[1])

# Function to evaluate the times to extinction of a set of time series
# Inputs:
#   seriesObj: object of the form 
#     [...,
#       {
#         timeLabel: 1D-array-like
#         seriesLabel: 1D-array-like
#       },
#     ...]
#     it stores the results of each repetition of the dynamics
#   timeLabel (optional): string
#   seriesLabel (optional): string
#   tol (optional): tolerance with which to evaluate the proximity of the values to 0
#
# Output: object of the form
#   { 
#     'times' : list, (values of the extinction times of the absorbing time series)
#     'mean' : None or float, (None: no time series is absorbing; float: average of the extinction times)
#     'std' : None or float, (None: no time series is absorbing; float: standard deviation of the extinction times)
#     'mean' : float, (fraction of absorbing time series)
#   }
def extinctionTimes(seriesObj,timeLabel='time',seriesLabel='resource',tol=1e-3):
    # Calculate the extinction times for each time series
    tExtVec = []
    for obj in seriesObj:
        tExt = extinctionTime_single(obj[timeLabel],obj[seriesLabel],tol)
        if tExt is not None:
            tExtVec.append(tExt)
    
    if tExtVec:
        tExtAvg = np.mean(tExtVec)
        tExtStd = np.std(tExtVec)
    else:
        tExtAvg = None
        tExtStd = None
    
    return {
        'times' : tExtVec,
        'mean' : tExtAvg,
        'std' : tExtStd,
        'fraction' : len(tExtVec)/len(seriesObj)
    }