import numpy as np
import os
from ..utils import _init_catalog_dict
"""
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
"""

class GFMModel(object):
    def __init__(self, comp_id, compname):
        self.comp_id=comp_id
        self.compname=compname
        self.splash_row = f'# -------- {self.compname} --------'
        
        self.modname =None
        self.par_dict =None
        self.deci=3
        
    def _update_vals(self):
        for i in self.par_dict.keys(): 
            print(i)
            
    def _parse_text(self, iobj, delim=','):
        if isinstance(iobj,list):
            if isinstance(iobj[0],str):
                plist=iobj
            else:
                plist=[str(i) for i in iobj]
            par_str=delim.join(plist)

        elif isinstance(iobj,bool):
            par_str=str(int(iobj))
        else:
            if iobj is None:
                par_str='none'
            else:
                par_str=str(iobj)
            
        return par_str

    def _setvals(self, inp):

        if isinstance(inp, list):
            fillval=np.nanmedian(inp)
            out=[np.round(i,self.deci) if np.isnan(i)==False else fillval for i in inp ]
        else:
            out=np.round(inp,self.deci)
        return out
        
    def dump_parstr(self):
        ostr_list=[self.splash_row, 
                   f'0)\t{self.modname}\t# Object type']

        for i in self.par_dict.keys(): 
            pd=self.par_dict[i]

            sind=str(pd['ind'])
            sval=self._parse_text(pd['val'])
            sfix=str(int(pd['fflag']))
            scom=pd['txt']

            rstr=f'{sind})\t{sval}\t{sfix}\t{scom}'
            

            # parse the setting
            #print(rstr)
            ostr_list.append(rstr)
            
        ostr_list+=['Z)\t0\t# Skip component','\n']
        
        return ostr_list

    def _parse_const(self, par, cdict):
        
        cid = self.comp_id
        ctype, cvals=cdict['ctype'], cdict['clims']

        # parse type of constrain
        if ctype=='to':
            ll,ul = cvals[0], cvals[1]
            ostr = f'{cid}\t{par}\t{ll}  {ctype}  {ul}'

        else:
            raise AssertionError
        return ostr
        
    

    def dump_const(self):
        
        out_cstr_list=[self.splash_row, 
                       '# Model Constrains']
        for i in self.par_dict.keys():         
            lims=self.par_dict[i]['lims']
            if len(lims)==0:
                pass
            else:
                for j  in lims:
                    out_cstr_list.append(self._parse_const(i,j))

        return out_cstr_list
                    
                
            



class Sersic(GFMModel):
    """
    Sersic model

    """
    
    def __init__(self, comp_id, compname):
        GFMModel.__init__(self, comp_id, compname)

        self.modname = 'sersic'
        self.par_dict={'x':{'ind':1, 'val':None, 'fflag': 1, 'lims': [], 'txt': '# x-position'},
                       'y':{'ind':2, 'val':None, 'fflag': 1, 'lims': [], 'txt': '# y-position'},
                       'mag': {'ind':3, 'val':22,   'fflag': 3, 'lims': [], 'txt': '# effective radius'}, 
                       're':  {'ind':4, 'val':10,   'fflag': 3, 'lims': [], 'txt': '# effective radius'}, 
                       'n':  {'ind':5, 'val':1.,    'fflag': 3, 'lims': [], 'txt': '# sersic index'},
                       'axr': {'ind':9, 'val':0.5,  'fflag': 1, 'lims': [], 'txt': '# axis ratio'},
                       'pa':  {'ind':10,'val':0,    'fflag': 1, 'lims': [], 'txt': '# PA'}}

        
        
    def setup_pars(self, label, segmap, data_dict):
        mag_list=[]
        for i, key in enumerate(data_dict):
            ddict=data_dict[key]
            ref_frame=bool(ddict['ref_frame'])
            band_dict=_init_catalog_dict(ddict['catalog'])

            obj_dict=band_dict[label]
            if ref_frame:
                # set the values
                self.par_dict['x']['val'] = self._setvals(obj_dict['xcentroid'])
                self.par_dict['y']['val'] = self._setvals(obj_dict['ycentroid'])
                self.par_dict['axr']['val'] = self._setvals(obj_dict['semiminor_sigma']/obj_dict['semimajor_sigma'])
                self.par_dict['pa']['val'] = self._setvals(90- obj_dict['orientation'].value)

                 # set the limits 
                xlim=self._setvals([obj_dict['bbox_xmin'],  obj_dict['bbox_xmax']])
                ylim=self._setvals([obj_dict['bbox_ymin'],  obj_dict['bbox_ymax']])
                dx=obj_dict['bbox_xmax']-obj_dict['bbox_xmin']
                dy=obj_dict['bbox_ymax']-obj_dict['bbox_ymin']
                rpix = self._setvals(np.sqrt(dx*dx + dy*dy))
                
                self.par_dict['x']['lims'].append({'ctype': 'to', 'clims':xlim})
                self.par_dict['y']['lims'].append({'ctype': 'to', 'clims':ylim})
                self.par_dict['re']['lims'].append({'ctype': 'to', 'clims': [0.3, rpix]})
                self.par_dict['n']['lims'].append({'ctype': 'to', 'clims': [0.3, 8]})
                
                
            # for now, use kronmag, next update use psf_mag
            mag_list.append(obj_dict['kron_mag'])
       
        # clean up nans
        self.par_dict['mag']['val'] = self._setvals(mag_list)
        self.par_dict['mag']['lims'].append({'ctype': 'to', 'clims':[0,40]})
            
    
class Pointsource(GFMModel):
    """
    Pointsource model

    """
    
    def __init__(self, comp_id, compname):
        GFMModel.__init__(self, comp_id, compname)

        self.modname = 'psf'
        self.par_dict={'x':{'ind':1, 'val':None, 'fflag': 1, 'lims': [], 'txt': '# x-position'},
                       'y':{'ind':2, 'val':None, 'fflag': 1, 'lims': [], 'txt': '# y-position'},
                       'mag': {'ind':3, 'val':22,   'fflag': 3, 'lims': [], 'txt': '# effective radius'},}

    def setup_pars(self, label, segmap, data_dict, limits=[], boundwith=None):
        
        mag_list=[]
        for i, key in enumerate(data_dict):
            ddict=data_dict[key]
            ref_frame=bool(ddict['ref_frame'])
            band_dict=_init_catalog_dict(ddict['catalog'])

            obj_dict=band_dict[label]
            if ref_frame:
                # set the values 
                self.par_dict['x']['val'] = self._setvals(obj_dict['xcentroid'])
                self.par_dict['y']['val'] = self._setvals(obj_dict['ycentroid'])

                # set the limits 
                xlim=self._setvals([obj_dict['bbox_xmin'],  obj_dict['bbox_xmax']])
                ylim=self._setvals([obj_dict['bbox_ymin'],  obj_dict['bbox_ymax']])
                self.par_dict['x']['lims'].append({'ctype': 'to', 'clims':xlim})
                self.par_dict['y']['lims'].append({'ctype': 'to', 'clims':ylim})
                
            # for now, use kronmag, next update use psf_mag
            mag_list.append(obj_dict['kron_mag'])

        # clean up nans
        # set values and limits 
        self.par_dict['mag']['val'] = self._setvals(mag_list)
        self.par_dict['mag']['lims'].append({'ctype': 'to', 'clims':[0,40]})

        # overwrite limits

class Sky2D(GFMModel):
    def __init__(self, comp_id, compname):
        GFMModel.__init__(self, comp_id, compname)

        self.modname = 'sky'
        self.par_dict={'bkg': {'ind':1, 'val':0, 'fflag': 1, 'lims': [], 'txt': '# average background '},
                        'dx': {'ind':2, 'val':0, 'fflag': 1, 'lims': [], 'txt': '# x-position'},
                        'dy': {'ind':3, 'val':0, 'fflag': 1, 'lims': [], 'txt': '# y-position'},}

    def setup_pars(self, label, segmap, data_dict, limits=[], boundwith=None):
        self.par_dict['bkg']['val'] = [0 for i in list(data_dict.keys())]
            
            
         
            
        














        