import os
import warnings
import pickle
#import astrophot as ap
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, interpolate_replace_nans, Gaussian2DKernel, Tophat2DKernel
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, MinMaxInterval, LinearStretch
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog, deblend_sources, SegmentationImage

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

plt.rcParams['font.size'] = 20
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.minor.visible'] = True




def fwhm_calc_gauss(indata, window=5):
    window = int(window)
    try:
        xcen, ycen = [int((i-1)/2) for i in indata.shape]

        crop_data = indata[xcen-window:xcen+window+1,
                           ycen-window:ycen+window+1]
        yi, xi = numpy.indices(crop_data.shape)
        fit = fitting.LevMarLSQFitter()
        mod = models.Gaussian2D(amplitude=crop_data.sum(),
                                x_mean=window, x_stddev=2,
                                y_mean=window, y_stddev=2)

        fitted_model = fit(mod, xi, yi, crop_data)
        fitted_image = fitted_model(xi, yi)
        # plt.imshow(crop_data-fitted_image)

        xfwhm = fitted_model.x_stddev.value*2.355
        yfwhm = fitted_model.y_stddev.value*2.355
    except:
        xfwhm, yfwhm = -99, -99

    return (xfwhm, yfwhm)


def calmag(zp,f,ferr):
    mag=zp-2.5*np.log10(f)
    magerr=2.5*ferr/f/np.log(10)
    return mag, magerr
    

class DataPrep:
    def __init__(self, dataset_dict, targ_name='aspmfit', targ_coord=None,
                 do_crop=False, cropfac=3, cropmin=5,  pxpad=10, **kwargs):
        
        # input parameters
        self.targ_name=targ_name
        self.targ_coord=targ_coord
        self.dataset_dict=dataset_dict    
        self.do_crop=do_crop
        self.cropfac=cropfac
        self.cropmin=cropmin
        self.pxpad=pxpad
        self.kwargs=kwargs
    
        
        # data properties
        self.n_ima=len(self.dataset_dict)
        self.band_names = list(self.dataset_dict.keys())
        
        self.refband_name=self.band_names[0]
        self.refband_dict=self.dataset_dict[self.refband_name]
        
        for i in self.band_names:
            if self.dataset_dict[i]['ref_frame']:
                self.refband_dict=self.dataset_dict[i]
                self.refband_name=i
        
        # initate empty attributes
        self.ref_seg=None
        
        
        print('--------------- Multiband Astrophot Setup ---------------')
        print(f'Received {self.n_ima} bands, including',','.join(self.band_names))
        print(f'Using {self.refband_name} as reference band')
        
        
    def _load_image(self, data_dict):
        ima_hdu=fits.open(data_dict['ima_path'])
        unc_hdu=fits.open(data_dict['unc_path'])
        unc_type=data_dict['unc_type'].lower().strip()
        
        #  fetch data
        ima_data=ima_hdu[data_dict['ima_hdu']].data
        ima_head=ima_hdu[data_dict['ima_hdu']].header
        unc_data=unc_hdu[data_dict['unc_hdu']].data
        
        # load wcs
        ima_wcs=WCS(ima_head)
        
        # parse uncertaintes to correct type
        if unc_type=='sigma':
            unc_data=unc_data*unc_data
        elif unc_type=='var':
            unc_data=unc_data
        else:
            raise ValueError('Unknown type of Uncertainty image')
                
        return ima_data, unc_data, ima_wcs
    
    def _load_psf_model(self, data_dict):
        psf_samp=float(data_dict['psf_samp'])
        psf_data=fits.getdata(data_dict['psf_path'],0)
        psf_data = psf_data.astype(float)
        #psf_model=ap.image.PSF_Image(data=psf_data, pixelscale=psf_samp)
        return psf_data
        
    
    def _background_subtract(self, ima, bkg_boxsize=32, bkg_filtersize=5, bkg_subtype='local', **kwargs):
        
        bs,fs=bkg_boxsize, bkg_filtersize
        bkg_obj = Background2D(ima, (bs, bs), filter_size=(fs, fs),
                               bkg_estimator=MedianBackground())
        
        rms = bkg_obj.background_rms
        
        if bkg_subtype == 'local':
            bkg = bkg_obj.background
        elif bkg_subtype == 'global':
            bkg = bkg_obj.background_median
        else:
            bkg = bkg_obj.background_median
            
        bks=ima-bkg
        
        return bks, bkg, rms
    
    
    def _convolve_image(self, bks, conv_kern='tophat', conv_kernfwhm=5, conv_kernsize=9, **kwargs):
        
        # choose kernel
        if conv_kern.lower()=='tophat':
            kernel= Tophat2DKernel(conv_kernsize)
        else:
            assert int(conv_kernsize)<int(conv_kernfwhm), 'Kernel size must be larger than FWHM'
            kernel= make_2dgaussian_kernel(conv_kernfwhm, size=conv_kernsize)
        # run convolution
        convolved_data = convolve(bks, kernel)
        return convolved_data
    
    def _create_segmap(self, conv, rms, det_sigma=3, det_minpx=9, **kwargs):
        
        threshold = det_sigma*rms
        # create segmentation map
        segment = detect_sources(conv, threshold, npixels=det_minpx, connectivity=8)
        
        return segment
    
    def _perform_deblend(self, conv, seg, dbl_npix=25, dbl_nlevel=32, dbl_contrast=0.05, **kwargs):
        
        # perform deblending
        segm_deblend = deblend_sources(conv, seg,
                                       npixels=dbl_npix, nlevels=dbl_nlevel,
                                       contrast=dbl_contrast,
                                       progress_bar=False)
        
        return segm_deblend

    
    def _source_detection(self, imaset, segobj=None, phot_bkgwidth=25, phot_kronpars=(2.5,1.4,0)):
        
        bks, rms, conv = imaset[2], imaset[4], imaset[5]
        
        #conv=self._convolve_image(bks, **self.kwargs)
        if segobj is None:
            # create new seg map for measurements
            segobj=self._create_segmap(conv, rms, **self.kwargs)
            segdbl=self._perform_deblend(conv, segobj, **self.kwargs)
        else:
            # assume the segmentation map provided has already been deblended
            segdbl=segobj
                
        catalog = SourceCatalog(bks, segdbl, convolved_data=conv, 
                                error=np.sqrt(imaset[1]), background=imaset[3],
                                wcs=imaset[-1], localbkg_width=phot_bkgwidth, 
                                kron_params=phot_kronpars)
        
        return catalog, segdbl
    
        
    # process functions
    def _process_image(self, data_dict):
        ima, var, ima_wcs=self._load_image(data_dict)
        # subtract background define rms
        bks, bkg, rms=self._background_subtract(ima, **self.kwargs)
        conv=self._convolve_image(bks, **self.kwargs)
        
        #print(ima_wcs)
        ima_wcs=self._recenter_wcs(ima_wcs)
        #print(ima_wcs)
        
        return [ima, var, bks, bkg, rms, conv, ima_wcs ]
    
    
    def _process_crop(self, imaset, size=3):
        
        ori_wcs = imaset[-1]
        
        pcen=[i/2 for i in imaset[0].shape]
        
        if self.targ_coord is None:
            ccen=tuple(float(i) for i in ori_wcs.pixel_to_world_values(pcen[1]-0.5, pcen[0]-0.5))
        else:
            ccen=self.targ_coord
            
        pxscale = proj_plane_pixel_scales(ori_wcs)[0]*3600
        assert int(self.cropmin/pxscale) > 0, 'Minimum cropsize is less than 1 pixel, please change'
        
        # define coordinate center and cropsize
        center=SkyCoord(ra=ccen[0]*u.degree, dec=ccen[1]*u.degree, frame='icrs')
        orisize = imaset[0].shape[0]*pxscale # original size unit arcsecs        
        asize = orisize/self.cropfac # cropped size in arcsec
        # asize = size # cropped size in arcsec, incase force the cropsize

        if asize < self.cropmin:
            csize=int(self.cropmin/pxscale)*u.pixel
        else:
            csize=int(asize/pxscale)*u.pixel
            
        # run crop
        ima_cr = Cutout2D(imaset[0], center, (csize,csize), wcs=ori_wcs, mode='partial')
        unc_cr = Cutout2D(imaset[1], center, (csize,csize), wcs=ori_wcs, mode='partial')
        bks_cr = Cutout2D(imaset[2], center, (csize,csize), wcs=ori_wcs, mode='partial')
        bkg_cr = Cutout2D(imaset[3], center, (csize,csize), wcs=ori_wcs, mode='partial')
        rms_cr = Cutout2D(imaset[4], center, (csize,csize), wcs=ori_wcs, mode='partial')
        cnv_cr = Cutout2D(imaset[5], center, (csize,csize), wcs=ori_wcs, mode='partial')
        
        ima_wcs=self._recenter_wcs(ima_cr.wcs)
        outset = [ima_cr.data, unc_cr.data, bks_cr.data, 
                  bkg_cr.data, rms_cr.data, cnv_cr.data, ima_wcs]
        
        return outset

    def _recenter_wcs(self, inwcs):
        outwcs=inwcs.copy()
        
        new_cen=-1*(inwcs.wcs.crpix)
        outwcs.wcs.crpix=[0,0]
        outwcs.wcs.crval=inwcs.pixel_to_world_values(*new_cen)
        
        return outwcs

    def _create_datamask(self, ima, err):

        tmask=np.zeros(ima.shape)
        tmask=tmask.astype(bool)
        tmask[np.isnan(ima)] = True
        tmask[np.isnan(err)] = True
        tmask[np.isinf(ima)] = True
        tmask[np.isinf(err)] = True
        tmask[ima==0] = True
        tmask[err==0] = True
        
        return tmask
        
    
    def run_init(self):
        # always nuke()
        self.nuke()
        
        print('--------------- Initiate Dataset ---------------')
        # fetch reference image first
        print('1. Creating Reference Images')
        
        if self.do_crop:
            temp_ref=self._process_image(self.refband_dict)
            self.ref_imaset=self._process_crop(temp_ref)
        else: 
            self.ref_imaset=self._process_image(self.refband_dict)
        
        # do source detection in the reference images
        self.ref_catobj, self.ref_seg=self._source_detection(self.ref_imaset)
        self.ref_wcs = self.ref_imaset[-1]
        # calculate additional photometry
        self.ref_cattable=self.ref_catobj.to_table()
        self.ref_cattable['kron_mag']=self.refband_dict['zeropoint']-2.5*np.log10(self.ref_cattable['kron_flux'])
        self.ref_cattable['kron_magerr']=(self.ref_cattable['kron_fluxerr']/self.ref_cattable['kron_flux'])/np.log(10)
        
        
        #self.ref_cattable['PA_deg']=(90.-self.ref_cattable['orientation'].value)
        #self.ref_cattable['q']=(self.ref_cattable['semiminor_sigma']/self.ref_cattable['semimajor_sigma']).value
        # define target ID
        
        if self.targ_coord is None:
            ycen,xcen=(int(i/2) for i in self.ref_seg.data.shape)    
        else:
            xcen,ycen=self.ref_wcs.world_to_pixel_values(*self.targ_coord)
            
        self.ref_targetid=self.ref_seg.data[int(ycen),int(xcen)]
        
        # loop through and execute
        print('2. Process individual frames')
        
        
        for i in self.dataset_dict.keys():
            idict=self.dataset_dict[i]
            
            temp_dict = {}
            temp_dict['ref_frame']=idict['ref_frame']
            temp_dict['zeropoint']=idict['zeropoint']
            temp_dict['psf_samp']=idict['psf_samp']
            temp_dict['psf_data']=self._load_psf_model(idict)
                        
            if self.do_crop:
                temp_mes=self._process_image(idict)
                mea_imaset=self._process_crop(temp_mes)
            else:
                mea_imaset=self._process_image(idict)
                
            temp_dict['ima_data']=mea_imaset[2]
            temp_dict['var_data']=mea_imaset[1]
            temp_dict['mask']=self._create_datamask(temp_dict['ima_data'], temp_dict['var_data'])
            temp_dict['ima_wcs']=mea_imaset[-1]

            # perform "forced" source cataloging of the bands
            scat, _=self._source_detection(mea_imaset, segobj=self.ref_seg)
            stab=scat.to_table()
            
            # calculate mags
            stab['kron_mag'],stab['kron_magerr']=calmag(temp_dict['zeropoint'], 
                                                        stab['kron_flux'], 
                                                        stab['kron_fluxerr'])

            temp_dict['catalog']=stab
            
            if 'metadata' in list(idict.keys()):
                temp_dict['metadata']=idict['metadata']
            #temp_dict['ima_wcs']=self._recenter_wcs(mea_imaset[-1])
            
            
            self.ap_dataset[i]=temp_dict
            self.meas_imaset.append(mea_imaset)
        print('--------------- Done ---------------')
            
    def nuke(self):
        
        # references
        self.ref_imaset=None
        self.ref_catobj=None
        self.ref_seg=None
        self.ref_cattable=None
        
        self.meas_imaset = []
        self.ap_dataset = {}


    def dump_dataset(self, outpath):
        datapack={'target_name': self.targ_name,
                  'target_coord': self.targ_coord,          
                  'dataset': self.ap_dataset, 
                  'catalog': self.ref_cattable,
                  'segmap': self.ref_seg,
                  'segwcs': self.ref_wcs,
                  'target_id': self.ref_targetid}
        with open(outpath, 'wb') as f:
            pickle.dump(datapack, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

          

        
            
    def _imascale(self,ima, interval_set='log', a=1000):
        
        interval_set=interval_set.lower().strip()
        if interval_set=='log':
            interval, stretch=MinMaxInterval(), LogStretch(a=a)
        else:
            interval, stretch=ZScaleInterval(), LinearStretch()
            
        norm=ImageNormalize(ima, interval=interval, stretch=stretch)
        return norm
            
    def plot_reference(self, fs=8, plotseg=False, plotfile='./detection.pdf',
                       saveplot=False, showplot=True, fontsize=25, cmap='gray_r', det_ec='k', det_lw=1.5,
                       target_marker='o',target_markersize=500 , legend_loc=4):
        
        ref_wcs = self.ref_imaset[-1]
        cat = self.ref_catobj
        norm=self._imascale(self.ref_imaset[2])
        
        if plotseg:
            fig=plt.figure(figsize=(fs*2,fs+0.25), dpi=75)
            ax1=fig.add_subplot(121, )
            ax2=fig.add_subplot(122, sharex=ax1,sharey=ax1)
            ax2.imshow(self.ref_seg, cmap=self.ref_seg.cmap,
                       interpolation='nearest', origin='lower')
            ax2.set_title('Segmentation', fontsize=fontsize)
        else:
            fig=plt.figure(figsize=(fs,fs), dpi=75)
            ax1=fig.add_subplot(111)
        
        ax1.set_title('Data', fontsize=fontsize)
        ax1.imshow(self.ref_imaset[2], norm=norm, cmap=cmap, origin='lower')
        cat.plot_kron_apertures(ax=ax1, color=det_ec, lw=det_lw)
        
        if self.targ_coord is not None:
            tc=ref_wcs.world_to_pixel_values(*self.targ_coord)
            ax1.scatter(tc[1], tc[0], marker=target_marker, ec='crimson', 
                        fc='None', s=target_markersize, label='Target')
            ax1.legend(loc=legend_loc, fontsize=fontsize-2)
            
            
        ax1.set_xlim(0, self.ref_imaset[2].shape[0])
        ax1.set_ylim(0, self.ref_imaset[2].shape[1])
        fig.suptitle(self.targ_name, fontsize=fontsize)
        fig.tight_layout()
        
        if saveplot:
            plt.savefig(plotfile, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.clf()
            
    def plot_measurements(self, fs=4, ncols=3, plotcat=True, plotfile='./measurements.pdf', 
                          saveplot=False, showplot=True, fontsize=18, cmap='gray_r', det_ec='k',det_lw=1.5,
                          target_marker='o', target_markersize=500, legend_loc=4):
    
        ndata=self.n_ima
        nrows=int(np.ceil(ndata/ncols))
        nsp=nrows*ncols
        ds_keys=list(self.ap_dataset.keys())
        
        ref_set=self.ref_imaset
        norm=self._imascale(ref_set[2])
        
        # source cats
        cat = self.ref_catobj
        
        fig, axs = plt.subplots(figsize=(ncols*fs, nrows*fs+0.25), nrows=nrows, ncols=ncols)
        axs=axs.ravel()
        for i in range(0, nsp):
            ax=axs[i]
            
            if i<ndata:
                band=ds_keys[i]
                ima_set = self.ap_dataset[band]
                ima_wcs = ima_set['ima_wcs']
                
                #norm=self._imascale(ima_set['ima_data'])
                ax.imshow(ima_set['ima_data'], norm=norm, origin='lower', cmap=cmap)
                
                ax.set_title(band, fontsize=fontsize)
                ax.set_xlim(0, ima_set['ima_data'].shape[0])
                ax.set_ylim(0, ima_set['ima_data'].shape[1])
                
                if plotcat:
                    cat.plot_kron_apertures(ax=ax, color=det_ec, lw=det_lw)
                    
                if self.targ_coord is not None:
                    tc=ima_wcs.world_to_pixel_values(*self.targ_coord)
                    ax.scatter(tc[1], tc[0], marker=target_marker, ec='crimson', fc='None', s=target_markersize,
                               label='Target')
                    ax.legend(loc=legend_loc, fontsize=fontsize-2)
                
            else:
                ax.imshow([[np.nan,np.nan],[np.nan,np.nan]], origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])  
            
            
        
        fig.suptitle(self.targ_name, fontsize=fontsize)
        fig.tight_layout()
        
        if saveplot:
            plt.savefig(plotfile, bbox_inches='tight')
        if showplot:
            plt.show()
        else:
            plt.clf()
        

