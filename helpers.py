import numpy as np
import matplotlib.pyplot as plt

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

def makePlots(plotsObj,tRep=False):
    if tRep:
        NRep = np.shape(plotsObj[1]['series'])[0]
        tVec = [plotsObj[0]['series'][k] for k in range(len(NRep))]
    else:
        tVec = plotsObj[0]['series']

    fig, axes = plt.subplots(1,2,figsize=(16,8))

    for ax,t,plotObj in zip(axes,tVec,plotsObj[1:]):
        ax.set_title(plotObj['name'])
        ax.plot(t,plotObj['series'],plotObj['color'], alpha=0.4)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.set_xlabel(plotObj['xlabel'])
        ax.set_ylabel(plotObj['ylabel'])
        ax.set_ylim(plotObj['yLims'])
        if plotObj['computeMean']:
            ax.hlines(np.mean(plotObj['series']),0,t[-1],linestyles='dashed',color=plotObj['color'],alpha=1)

    return fig, axes

def evalContextVar(names, context):
    return [eval(name, {}, context) for name in names]