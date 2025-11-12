import os
import numpy as np
import time


def _init_catalog_dict(incat):
        cat_dict={}

        rev_cat=incat

        # calculate additiona quantities
        cols=list(rev_cat.keys())
        for row in rev_cat:
            window=row.get('label')
            data=[row.get(j) for j in cols]
            cat_dict[window]=dict(zip(cols,data))
            
        return cat_dict
    
def _print_time_used(intime):
    dt=time.time()-intime
        
    if int(dt/60)>=1:
        if int(dt/3600)>=1:
            pt, tunit=dt/3600, 'hr'
        else:
            pt, tunit=dt/60, 'minutes'
    else:
        pt, tunit=dt, 'second'
            
    tstr='Finished, took {:.2f} {}.'
    print(tstr.format(pt,tunit))