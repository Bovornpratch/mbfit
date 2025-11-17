import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from astropy.io import fits


class DataAnalysis:
    def __init__(self, res_pkl):

        # set attributes
        self.res_pkl = res_pkl
        self.res_dir = os.path.dirname(self.res_pkl)
        self.dataset = self._load_results()

        self.sc_ima=os.path.join(self.res_dir, self.dataset['sc_ima'])
        self.bf_ima=os.path.join(self.res_dir, self.dataset['bf_ima'])

        
        self.obs_bands= self.dataset['config_dict']['band_list']
        self.obs_wavelength= self.dataset['config_dict']['wavelength_list']
        self.bfmod = self._parse_results()
        self.stats = self._fetch_summary_stats()
        
    
    def _fetch_file(self, indir, flag):
        gstr = os.path.join(indir,flag)
        flist=glob.glob(gstr)
        
        if len(flist)==0:
            of='None'
        elif len(flist)==1:
            of=flist[0]
        else:
            raise AssertionError('Too many files found')    

        return of
    
    def _load_results(self):
        with open(self.res_pkl, "rb") as f:
            respkl = pkl.load(f)
        return respkl
        
    def _collect_model(self, infile):
        hdulist=fits.open(infile)
        #model_map = model_map
        hdu_dict={}
        for i in range(1, len(hdulist)) :
            head=hdulist[i].header
            imatype, bandname = head['EXTNAME'].split('_')
            if imatype != 'MODEL':
                pass
            else:
                hdu_dict[bandname]={'header': head}
        return hdu_dict

    def _str2num(self, instr):
        
        instr=instr.replace(' ', '')
        instr=instr.replace('*', '')
        
        meas, uncr = instr.split('+/-')
        # check blocks 
        # do nothing for now
        return (float(meas), float(uncr))

    def _parse_bestfit(self, cnum, ctype, band, header):
        band=band.upper()
    
        if ctype=='sersic':
            basekey = ['XC', 'YC', 'MAG', 'Re', 'n', 'AR', 'PA']
        elif ctype=='psf':
            basekey = ['XC', 'YC', 'MAG']
        else:
            raise AssertionError('Unknown model during bestfit parse')

        pkeys = [f'{cnum}_{i}_{band}' for i in basekey]
        pvals,puncs = zip(*[self._str2num(header[i]) for i in pkeys])

        #return pkeys, pvals, puncs
        return basekey, pvals, puncs
    
    def _parse_results(self):
        model_dict = self.dataset['config_dict']['model_map'].copy()
        band_dict = self._collect_model(self.bf_ima)

        model_names = [i for i in model_dict.keys() if i!='sky']
        band_names = list(band_dict.keys())

        bfres_dict = {}
        for tid in model_names:
            mlist=model_dict[tid]['mod_index']
            comp_dict={}
            for mod in mlist:
                comp_name = f'COMP_{mod}'

                keys, vals, uncs = [], [], []
                for band in band_names:
                    bhead = band_dict[band]['header']

                    comp_type = bhead[comp_name]
                    res=self._parse_bestfit(mod, comp_type, band, bhead)

                    keys.append(res[0])
                    vals.append(res[1])
                    uncs.append(res[2])

                # set values 
                #meow[0]
                pars = keys[0]
                vals, uncs = list(zip(*vals)), list(zip(*uncs))
                meas=[ {'val':list(vals[i]), 'unc':list(uncs[i])} for i in range(0, len(pars))]

                comp_dict[comp_name]={'ctype':comp_type, 'pars': dict(zip(pars,meas))}

                #break
            bfres_dict[tid] = comp_dict
                
        return bfres_dict

    def _fetch_summary_stats(self):
        band_dict = self._collect_model(self.bf_ima)
        band_names = list(band_dict.keys())
        stats_dict={}
        
        for band in band_names:
            bhead = band_dict[band]['header']
            
            stats_dict[band]={'CHISQ':bhead['CHISQ'],
                              'NFREE':bhead['NGOOD'],
                              'NFREE':bhead['NFREE'],
                              'NFIX':bhead['NFIX'],
                              'CHI2NU':bhead['CHI2NU'],
                              'NPIX':bhead['NAXIS1']*bhead['NAXIS2'],
                              'wavelength': bhead[f'WL_{band}']}
            #print(bhead)
        return stats_dict

    def dump_summary(self, outfile):
        outset={'ObsBand':self.obs_bands,
                'ObsWave':self.obs_wavelength, 
                'Stats': self.stats, 
                'BestfitMod': self.bfmod,}
    
        with open(outfile, 'wb') as f:
            pkl.dump(outset, f)
            

 
        