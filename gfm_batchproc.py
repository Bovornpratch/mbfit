import os
import glob
import sys
import warnings
import json
import numpy as np
import pickle as pkl 
from time import time
from astropy.utils.exceptions import AstropyWarning
from multiprocessing import Pool
from .gfm_single import MBfitGalfitM
from .utils import  _print_time_used

# internal settins
warnings.simplefilter('ignore', category=AstropyWarning)



# read the file paths
class GalfitmBatchProc:
    def __init__(self, inputfile, outdir='./test',  
                 ngal=1, npsf=0,
                 showplots=True,
                 logdir='./logs'):
        
        self.inputfile=inputfile
        self.outdir=outdir
        self.ngal=ngal
        self.npsf=npsf
        self.showplots=showplots
        self.logdir=logdir

        # process
        self.process_file = self._fetchdata()
                
    
    def _fetchdata(self):
        with open(self.inputfile, 'r') as f:
            templist=[i.strip('\n') for i in f.readlines()]
            
        flist=[]
        for i in templist:
            eflag=os.path.exists(i)
            if eflag:
                flist.append(i)
                
        print('Found ', len(flist))

        return flist

    def _read_pkl(self, inp):
        with open(inp, "rb") as f:
            out = pkl.load(f)
        return out

    def fit_single(self, indata, execute):

        # read the dataset dictionary
        indict=self._read_pkl(indata)

        # get some meta data
        tname=indict['target_name']
        tcoord=indict['target_coord']

        # init results dict
        res_dict = {'Name': tname, 'Failed':0, 
                    'stage_track': {'InitMod': 0, 'WriteInput': 0, 'RunFit': 0}}

        # set and create output dir
        toutdir=os.path.join(self.outdir, tname)
        os.makedirs(toutdir, exist_ok=True)
        
        # use center coordinats for now
        # nothing to do

        # set up the analysis object
        mbfit_obj=MBfitGalfitM(indict, outdir=toutdir,
                                   target_name=indict['target_name'])
        try:
            
            setup_pdf=os.path.join(toutdir, tname+'_setup.pdf')
            mbfit_obj.plot_setup(plotfile=setup_pdf, saveplot=True, 
                                 showplot=self.showplots)
            
        except:
            res_dict['Failed']=1
            res_dict['stage_track']['InitMod']=1

        # create the config
        try:
            cfg_dict=mbfit_obj.init_config(writefiles=execute, 
                                           ngal=self.ngal, npsf=self.npsf)
        except:
            res_dict['Failed']=1
            res_dict['stage_track']['WriteInput']=1
        
        # run the fit
        
        if execute is False:
            print('Done')
        else:
            try:
                res = mbfit_obj.execute_fit(cfg_dict)
            except:
                res_dict['Failed']=1
                res_dict['stage_track']['RunFit']=1

        # clean up
        return res_dict

    def execute_fitting(self, ncpu=1, execute_run=False, save_summary=False):

        # create the directories 
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)
        timer = time()
        # create args
        argslist=[]
        for file in self.process_file:
            cfg_row=(file,execute_run)
            argslist.append(cfg_row)
            
        #argslist=argslist[63:65]
        argslist=argslist[0:10]
        
        # execute the fit
        print('Running the fit')
        if ncpu==1:
            # run loop
            results = [self.fit_single(*args) for args in argslist]
        else:
            with Pool(processes=ncpu) as pool:
                results=pool.starmap(self.fit_single, argslist)

        sdict={
              'FailedObjects':[],
              'stage_track':{},}

        outsum=0
        for i in results:
            tname, failed=i['Name'],i['Failed']
            sdict['stage_track'][tname]=i
        
            if failed==1:
                sdict['FailedObjects'].append(tname)
                outsum+=1

        if save_summary:
            outfile=os.path.basename(self.inputfile).split('.')[0] + '.json'
            outpath=os.path.join(self.logdir, outfile)
            
            with open(outpath, "w") as f:
                json.dump(sdict, f)

        ptext = f'Finished all the fitting, {outsum} sources failed'
        if outsum>0:
            ptext+='\n'
            for i in sdict['FailedObjects']:
                ptext+= i+'\n'
        
        print('---------------------------------')
        print(ptext)
        _print_time_used(timer)
        print('---------------------------------')
        


        return sdict

        