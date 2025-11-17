import os 
import time
import subprocess
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import binary_dilation
#from scripts.mbfit.utils import _init_catalog_dict
from .galfitm.models import Sersic, Pointsource, Sky2D
from .utils import _print_time_used, _init_catalog_dict

from astropy.io  import fits
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch, MinMaxInterval, ZScaleInterval


class MBfitGalfitM:
    def __init__(self, dataset, outdir='./mbfit/galfitm/', 
                 target_name='mbfit', target_id=None, target_coord=None,
                 fit_mask_type='local', fit_mask_expand=10, fit_expand=1.2, fit_pxpad=5, 
                 fit_edgepad = 3, fit_convboxfac=2.5,
                 default_refband = -1):

        if isinstance(dataset, str):
            with open(dataset, "rb") as f:
                self.dataset = pkl.load(f)
        elif isinstance(dataset, dict):
             self.dataset= dataset
        else:
            raise AssertionError('Unknown data format')

        print("-------- Initiate MEOWWWWW --------")
        # set segmentation attributes
        self.segmap=self.dataset['segmap']
        self.segwcs=self.dataset['segwcs']
        self.seg_ima = self.segmap.data

        # catalog objects
        self.catalog=self.dataset['catalog']
        self.catalog_dict = _init_catalog_dict(self.catalog)

        # additional image attributes
        self.ysize,self.xsize = self.seg_ima.shape
        self.ycen,self.xcen=(int(i/2) for i in self.seg_ima.shape)

        # sets source attributes
        self.target_name = target_name
        if target_id is None:
            self.target_id=self.seg_ima[self.ycen,self.xcen]
        else:
            self.target_id=int(target_id)
            
        # read and check data
        self.dataset_dict = self._data_qc(self.dataset['dataset'])

        # check reference bands 
        self.bandnames = list(self.dataset_dict.keys())
        self.refband_name = None
        for i in self.bandnames:
            if self.dataset_dict[i]['ref_frame']:
                self.refband_name=i

        # set reference band in case the reference band is not availible    
        if self.refband_name is None:
            self.refband_name = self.bandnames[default_refband]
            self.dataset_dict[self.refband_name]['ref_frame']=1

        # define reference band dictionary for completeness
        self.refband_dict = self.dataset_dict[self.refband_name]
        
        # set fitting windows parameters
        self.fit_mask_type=fit_mask_type
        self.fit_mask_expand=int(fit_mask_expand)
        self.fit_expand=fit_expand 
        self.fit_pxpad=fit_pxpad
        self.fit_edgepad = fit_edgepad
        self.fit_convboxfac = fit_convboxfac

        # initiate the dataset
        self.det_windows=self._create_windows()
        self.fit_windows=self._optimize_windows()     
        self.mask=self._create_mask()

        # fitting model internal settings
        self.galmod_map = {'sersic': Sersic}
        
        # set I/O attributes
        self.outdir = outdir
        self.gfm_cfg = f'{self.target_name}_gfm.cfg'
        self.gfm_const = f'{self.target_name}_const.txt'

        # print the splash
        self._print_splash()
        self.timer=time.time()
    
    def _print_splash(self):
        dnames = [str(i) for i in self.det_windows.keys()]
        tnames = [str(i) for i in self.fit_windows.keys()]
        print(f"Bands to fit:{','.join(self.bandnames)}")
        print(f"Reference Band is:{self.refband_name}")
        print(f"Target is {self.target_name} : ID{self.target_id}")
        print('Detected  IDs: '+','.join(dnames))
        print('Will fit only IDs: '+','.join(tnames))
        

    def _data_qc(self, input_dict):
        tid = self.target_id
        seg = self.seg_ima
        cleaned_data = {}
        
        input_bands = list(input_dict.keys())
        print('Start Data QC stage')
        for i in input_bands:
            ddict = input_dict[i]
            temp_mask = ddict['mask']
        
            # data QC check
            mask_pix = temp_mask[seg==tid]
            mask_flag = np.any(mask_pix)
            
            # meow
            cflag=mask_flag

            if cflag:
                print(i, 'Failed QC')
            else:                
                print(i, 'Passed QC')
                cleaned_data[i] = ddict

        # check and raise assertion error if 
        # there is no data at all
        if len(cleaned_data.keys())==0:
            raise AssertionError('No data passed QC for analysis')
        else:
            pass
            
        return cleaned_data
    
        
    def _create_windows(self):

        ndet = len(self.segmap.labels)
        outdict = {}
        
        for i in range(0, ndet):
            obox,nbox =self.segmap.bbox[i].__dict__ , {}
            ysize,xsize = self.ysize,self.xsize
            dx, dy=0,0
            xp = self.fit_pxpad + dx
            yp = self.fit_pxpad + dy

            nbox['ixmin']= 0 if obox['ixmin']-xp < 0 else obox['ixmin']-xp
            nbox['ixmax']= xsize if obox['ixmax']+xp > xsize else obox['ixmax']+xp
            nbox['iymin']= 0 if obox['iymin']-yp < 0 else obox['iymin']-yp
            nbox['iymax']= ysize if obox['iymax']+xp > ysize else obox['iymax']+xp

            outdict[i+1] = nbox
            
        return outdict

    def _optimize_windows(self):

        fit_index=[self.target_id]
        if self.fit_mask_type.lower()=='local':
            fit_index+=self._fetch_segindex(self.det_windows[self.target_id])
            fit_index=list(set(fit_index))
    
        outdict = {key:val for key, val in self.det_windows.items() if key in fit_index}
            
        return outdict

    def _fetch_segindex(self, bounds):
        
        xb=[bounds['ixmin'], bounds['ixmax']]
        yb=[bounds['iymin'], bounds['iymax']]
        
        crop=self.segmap.data[yb[0]:yb[1]+1, xb[0]:xb[1]+1]
        #plt.imshow(crop, origin='lower')
        out_index=list(set(crop.flatten()))
        out_index = [i for i in out_index if i != 0]
        
        return out_index
        
    def _create_mask(self):
        # initial bounding box mask index
        det_list = list(self.det_windows.keys())
        fit_list = list(self.fit_windows.keys())
        mask_index=[i for i in det_list if i not in fit_list] 
        
        # flib bit mask
        init_mask=np.zeros(self.seg_ima.shape)
        for i in mask_index:
            init_mask[self.seg_ima==i]=1

        outmask = init_mask.astype(bool)
        outmask=binary_dilation(outmask, iterations=self.fit_mask_expand)
        
        return outmask

    def _merge_mask(self, basemask, mask_list):
        outmask=basemask.copy()
        for i in mask_list:
            outmask=basemask | i
    
        return outmask

    def _write_image(self, header, data, fpath):
        fpath=os.path.join(self.outdir, fpath)
        outhdu=fits.PrimaryHDU(data=data, header=header)
        outhdu.writeto(fpath, overwrite=True)

    def _write_to_textfile(self, ostr, file, mode='w'):
        fpath = os.path.join(self.outdir, file)
        with open(fpath, mode) as f:
            f.writelines([i+'\n' for i in ostr])
    
    def _parse_text(self, iobj, delim=','):
        if isinstance(iobj,list):
            if isinstance(iobj[0],str):
                plist=iobj
            else:
                plist=[str(i) for i in iobj]
            par_str=delim.join(plist)
            
        else:
            if iobj is None:
                par_str='none'
            else:
                par_str=str(iobj)
            
        return par_str

    def _contruct_config_head(self, cfg):
        ostr = [f'# GALFITM Parameter file for target={self.target_name}',
                '# CONFIG HEADER STARTS HERE',
                'A)\t'+self._parse_text(cfg['image_list']),
                'A1)\t'+self._parse_text(cfg['band_list']),
                'A2)\t'+self._parse_text(cfg['wavelength_list']),
                'B)\t'+self._parse_text(cfg['output_block']),
                'C)\t'+self._parse_text(cfg['sigma_list']),
                'D)\t'+self._parse_text(cfg['psf_list']),
                'E)\t'+self._parse_text(cfg['psfsamp']),
                'F)\t'+self._parse_text(cfg['mask_list']),
                'G)\t'+self._parse_text(cfg['const_file']),
                'H)\t'+self._parse_text(cfg['fitbounds'], delim=' '),
                'I)\t'+self._parse_text(cfg['convbox'], delim=' '),
                'J)\t'+self._parse_text(cfg['zp_list']),
                'K)\t'+self._parse_text(cfg['platescale'],' '),
                'O) regular', 'P)  0', 
                'U) '+self._parse_text(cfg['cheb_par'], delim=' '),
                'V) 0', 'W) default',
                '#CONFIGHEADER_END']
        
        return ostr

    def _prepare_config(self, fit_withconst=True, writefiles=False):

        # pixel sampling differences, can be put into loop for 
        # individual bands meow
        ref_wcs=self.refband_dict['ima_wcs']
        ref_pxs=proj_plane_pixel_scales(ref_wcs)*3600
        pxsamp = ref_pxs[0]/self.refband_dict['psf_samp']
        psfsize = self.refband_dict['psf_data'].shape
        
        # edge pad to ignore
        ep =  self.fit_edgepad
        
        # start initiation of galfit parameters and key words
        # always do stuff in local directory since the fitting 
        # stage will run in it with sub process
        
        cfg_dict = {'image_list': [],
                    'band_list': [], 'wavelength_list':[], 
                    'output_block': f'{self.target_name}_gfm_res.fits',
                    'sigma_list': [],
                    'psf_list': [], 'psfsamp':np.round(pxsamp,3),
                    'mask_list' : [],
                    'const_file': self.gfm_const if fit_withconst==True else None,
                    'fitbounds': [ep, self.xsize-ep, ep, self.ysize-ep],
                    'convbox': [int(i*self.fit_convboxfac) for  i in psfsize],
                    'zp_list': [],
                    'platescale': [np.round(i,3) for i in ref_pxs],
                    'npar_set': None, 
                    'param_str': None,
                    'const_str': None,
                    'ima_set': {},
                    }

        
        # dump out the images 
        for i in self.dataset_dict.keys():
            
            ddict=self.dataset_dict[i]
            meta=ddict['metadata']
            ima_head = ddict['ima_wcs'].to_header()
            unc_head = ddict['ima_wcs'].to_header()
            mas_head = ddict['ima_wcs'].to_header()
            psf_head = fits.header.Header()
            
            
            pprefix = f'{self.target_name}_{i}'
            oima, osig = pprefix+'_SCI.fits', pprefix+'_ERR.fits'
            opsf = pprefix+'_psf.fits'
            omas = pprefix+'_mask.fits'
            
            # append image list 
            cfg_dict['band_list'].append(i)
            cfg_dict['wavelength_list'].append(ddict['metadata']['wavelength'])
            cfg_dict['image_list'].append(oima)
            cfg_dict['sigma_list'].append(osig)
            cfg_dict['psf_list'].append(opsf)
            cfg_dict['mask_list'].append(omas)
            cfg_dict['zp_list'].append(ddict['zeropoint'])

            # create files for analysis
            band_mask = self._merge_mask(ddict['mask'], [self.mask])

            cfg_dict['ima_set'][i] = {'ima':ddict['ima_data'], 
                                      'unc':np.sqrt(ddict['var_data']),
                                      'mask':band_mask.astype(int)}

            if writefiles:
                # pixel mask, keep in loop in case we need to append 
                # several mask together, quite likely we need to do it
                

                # write out the data
                self._write_image(ima_head, ddict['ima_data'], oima)
                self._write_image(unc_head, np.sqrt(ddict['var_data']), osig)
                self._write_image(mas_head, band_mask.astype(int), omas)
                self._write_image(psf_head, ddict['psf_data'], opsf)

        # write the config
        return cfg_dict

    def _psf_mapping(self, map_psf_to):
        if isinstance(map_psf_to, list):
            psf_map=map_psf_to
        elif isinstance(map_psf_to, str):
            map_psf_to = map_psf_to.lower().strip()    
            if map_psf_to=='target':
                print('Map pointsource to target')
                psf_map = [self.target_id]
            elif map_psf_to=='all':
                print('Map pointsource to all object')
                psf_map = self.fit_windows
            else:
                raise AssertionError(f'Choose between "all" or "target"')
        else:
            raise AssertionError(f'Input of "map_psf_to" must be "all" or "target" or a list of target ids')
            
        return psf_map
        
    
    def init_config(self, writefiles=False,
                    ngal=1, npsf=0, map_psf_to='target', 
                    galmod='sersic', fit_withconst=True, add_sky=True, 
                    cheb_par=-1):

        # prepare output directory and checks in case 
        os.makedirs(self.outdir, exist_ok=True)
        cfg_fullpath=os.path.join(self.outdir, self.gfm_cfg)
        galmod = galmod.lower().strip()
        assert ngal+npsf>0, 'Come on, you gotta fit something...'        
        
        # check input model request with mapping
        if galmod not in list(self.galmod_map):
            raise AssertionError(f'{galmod} Model not in the current mapping')

        print('---------- Initiating Config ----------')
        print(f'Using {galmod} as galaxy model')
        print(f'Fitting {ngal} galaxy model per detection')
        
         # set mapping for pointsource model
        psf_map = self._psf_mapping(map_psf_to)
        
        # generate configuration
        cfg_dict=self._prepare_config(fit_withconst=fit_withconst, 
                                      writefiles=writefiles)
        cfg_dict['cheb_par']=cheb_par
        cfg_str_list=self._contruct_config_head(cfg_dict)
        
        # initiate the weird non-parameteic component
        # Will come back and check this again.

        
        # append parameters to this string
        # we force LM fit first, "V) 0"
        parstr_list, constr_list = [], []
        # now we create some objects

        mod_dict={}
        mod_id=1
        for i in self.fit_windows:
            compnum=1
            modlist = []
            complist = []
            
            # add the models number of stuff
            for j in range(0, ngal):
                Galmod = self.galmod_map[galmod]
                compname=f'ID{i}_mod_{galmod}_comp_{mod_id}'
                gmod = Galmod(mod_id, compname=compname)
                gmod.setup_pars(i,self.segmap, self.dataset_dict)
                parstr_list+=gmod.dump_parstr()
                constr_list+=gmod.dump_const()+['\n']

                modlist.append(mod_id)
                complist.append(compnum)
                
                mod_id+=1
                compnum+=1

            for j in range(0, npsf):
                if i not in psf_map:
                    pass
                else:
                    print(f'Adding PSF to ID{i}')
                    
                    compname=f'ID{i}_mod_psrc_comp_{mod_id}'
                    pmod = Pointsource(mod_id, compname=compname)
                    pmod.setup_pars(i,self.segmap, self.dataset_dict)
                    parstr_list+=pmod.dump_parstr()
                    constr_list+=pmod.dump_const()+['\n']
                
                    modlist.append(mod_id)
                    complist.append(compnum)
                
                    mod_id+=1
                    compnum+=1
                    
            mod_dict[i]={'mod_index': modlist}

        if add_sky:
            compname=f'ID{i}_mod_sky2d_comp_{0}'
            smod = Sky2D(mod_id, compname=compname)
            smod.setup_pars(i,self.segmap, self.dataset_dict)
            parstr_list+=smod.dump_parstr()

            mod_dict['sky']={'mod_index': [mod_id]}
            mod_id+=1
            compnum+=1
            
            
            

        # set model and const
        cfg_dict['param_str']=parstr_list
        cfg_dict['model_map']= mod_dict

        # flush everything to the config file
        print(f'Writing config file {self.gfm_cfg}')
        self._write_to_textfile(cfg_str_list+parstr_list, self.gfm_cfg, mode='w')

        if fit_withconst:
            print(f'Writing constrain file {self.gfm_const}')
            cfg_dict['const_str']=constr_list
            self._write_to_textfile(constr_list, self.gfm_const)
        
        return cfg_dict
    
   
    def execute_fit(self, cfgdict, niter=100, write_res=True, splash=False):

        # config files
        print('#----------------------------------------------#')
        print('# FITTING GO BRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR #')

        # set upt files
        res_file = cfgdict['output_block']
        band_file = res_file.replace('.fits', '.galfit.01.band')
        cheb_file = res_file.replace('.fits', '.galfit.01')
        logfile = res_file.replace('.fits', '.log')

        # best fit files
        band_bftxt = res_file.replace('res.fits', 'band_bfpar.txt')
        cheb_bftxt = res_file.replace('res.fits', 'cheb_bfpar.txt')
        
        self.timer=time.time()
        #"""
        subprocess.run(f'galfitm {self.gfm_cfg}', cwd=self.outdir,  capture_output=True, shell=True)
        _print_time_used(self.timer)

        # clean up
        subprocess.run(f'mv fit.log {logfile}', cwd=self.outdir, capture_output=True, shell=True)
        subprocess.run(f'mv {band_file} {band_bftxt}', cwd=self.outdir, capture_output=True, shell=True)
        subprocess.run(f'mv {cheb_file} {cheb_bftxt}', cwd=self.outdir, capture_output=True, shell=True)
        #"""
        # generate sub components
        temp_file = 'temp_file.fits'
        scomp_file = res_file.replace('.fits','_sc.fits')
        #"""
        subprocess.run(f'mv {res_file} {temp_file}', cwd=self.outdir, capture_output=True, shell=True)
        subprocess.run(f'galfitm -o3 {band_bftxt}', cwd=self.outdir,  capture_output=True, shell=True)
        subprocess.run(f'mv {res_file} {scomp_file}', cwd=self.outdir, capture_output=True, shell=True)
        subprocess.run(f'mv {temp_file} {res_file}', cwd=self.outdir, capture_output=True, shell=True)
        #"""
       
        bfres_path = os.path.join(self.outdir,res_file)
        outdata_set  = {'resdir': self.outdir, 
                        'catalog_dict': self.catalog_dict,
                        'refband': self.refband_name,
                        'det_windows': self.det_windows,
                        'fit_windows': self.fit_windows,
                        'proc_data':  self.dataset_dict,
                        'config_dict' : cfgdict,
                        'seg_data': self.seg_ima,
                        'bf_ima':res_file,
                        'sc_ima':scomp_file}

        if write_res:
            outfile = self.target_name+'_gfmr_result.pkl'
            outpath = os.path.join(self.outdir, outfile)
            
            with open(outpath, 'wb') as f:
                pkl.dump(outdata_set, f, 
                         protocol=pkl.HIGHEST_PROTOCOL)

        print('#----------------------------------------------#')

        return outdata_set

    def plot_setup(self, ncols=4, fs=3, fontsize=14, plotcat=True, plotfile='./detection.pdf', 
                   saveplot=False, showplot=True, ):

        ndata=len(self.bandnames)
        nrows=int(np.ceil(ndata/ncols))
        nsp=nrows*ncols

        fig, axs = plt.subplots(figsize=(ncols*fs, nrows*fs+0.25), nrows=nrows, ncols=ncols)
        axs=axs.ravel()


        for i in range(0, ndata):
            ax=axs[i]
            temp_band=self.bandnames[i]

            ddict=self.dataset_dict[temp_band]

            mask = ddict['mask'] | self.mask
            inv_mask = (~mask).astype(int)
            pdata=ddict['ima_data'].copy()
            
            norm=ImageNormalize(ddict['ima_data']*inv_mask,
                                stretch=LogStretch(), interval=MinMaxInterval())

            pdata[mask==True]=np.nan

            # Choose the color
            cmap = plt.cm.get_cmap("gray_r").copy()
            cmap.set_bad('w',1.)

            # plots
            ax.imshow(pdata,cmap=cmap, origin='lower', norm=norm)

            ax.set_title(temp_band, fontsize=fontsize)
            ax.set_xlim(0, ddict['ima_data'].shape[0])
            ax.set_ylim(0, ddict['ima_data'].shape[1])

            # plot the squares

            for det in self.fit_windows:
                cdict=self.catalog_dict[det]

                x,y= cdict['bbox_xmin'], cdict['bbox_ymin']
                dx=cdict['bbox_xmax']-cdict['bbox_xmin']
                dy=cdict['bbox_ymax']-cdict['bbox_ymin']

                rect = patches.Rectangle((x,y), dx, dy, linewidth=1,
                                         edgecolor='cyan', facecolor="none")
                ax.add_patch(rect)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])  

        fig.suptitle(self.target_name, fontsize=fontsize)
        fig.tight_layout()
        
        
        if saveplot:
            plt.savefig(plotfile, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.clf()
        

