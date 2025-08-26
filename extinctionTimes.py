import numpy as np

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

def extinctionTimes(seriesObj,timeLabel='time',resLabel='resource',tol=1e-3):
    # Calculate the extinction times for each time series
    tExtVec = []
    for obj in seriesObj:
        tExt = extinctionTime_single(obj[timeLabel],obj[resLabel],tol)
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