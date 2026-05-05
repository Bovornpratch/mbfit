# 
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch, MinMaxInterval, ZScaleInterval
from astropy.io import fits

def plot_setup(resdict, ncols=4, fs=3, fontsize=15, plotfile='./setup.pdf', 
               stdscale=5, saveplot=False, showplot=True, ):

    # fetch bestfit file
    bffile=os.path.join(resdict['resdir'], resdict['bf_ima'])
    indata=resdict['proc_data']
    
    config=resdict['config_dict']
    fb=config['fitbounds']
    imset=config['ima_set']
    segmap=resdict['seg_data'][fb[2]-1:fb[3],fb[0]-1:fb[1]]
    
    # read input fits file
    ima_hdu=fits.open(bffile)

    hdu_names=[i.name for i in ima_hdu] 
    band_list=[i.name.split('_') [1] for i in ima_hdu  if 'INPUT_' in i.name ]
    
    band_dict={}
    
    for band in band_list:
    
        inp_hdu=ima_hdu[hdu_names.index(f'INPUT_{band}')]
        res_hdu=ima_hdu[hdu_names.index(f'RESIDUAL_{band}')]
        mod_hdu=ima_hdu[hdu_names.index(f'MODEL_{band}')]

        unc_data=imset[band]['unc']
        msk_data=imset[band]['mask']
        
        unc_data=unc_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]
        msk_data=msk_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]
        
        band_dict[band]={'ima_data':inp_hdu.data,
                         'unc_data':unc_data,
                         'res_data':res_hdu.data,
                         'mod_data':mod_hdu.data,
                         'msk_data':msk_data,}

    refband=resdict['refband']
    refpix=band_dict[refband]['ima_data']
    norm=ImageNormalize(refpix, stretch=LogStretch(), interval=MinMaxInterval())
    
    # plot it
    ndata=len(band_list)
    nrows=int(np.ceil(ndata/ncols))
    nsp=nrows*ncols
    fig, axs = plt.subplots(figsize=(ncols*fs, nrows*fs+0.25), nrows=nrows, ncols=ncols)
    axs=axs.ravel()
    
    for i in range(0, len(band_list)):
        band = band_list[i]
        bdict=band_dict[band]

        ima=bdict['ima_data']
        unc=bdict['unc_data']
        res=bdict['res_data']
        msk=bdict['msk_data']
        
        nres=ima.copy()
        nres[msk>0]=np.nan
        
        #axs[i].imshow(ima, origin='lower', vmin=vmin, vmax=vmax, cmap='PuOr_r')
        axs[i].imshow(nres, origin='lower', norm=norm, cmap='gray_r')
        axs[i].set_title(band, fontsize=fontsize)
        axs[i].set_xlim(0, ima.shape[0])
        axs[i].set_ylim(0, ima.shape[1])

    fig.tight_layout()
        
    if saveplot:
        plt.savefig(plotfile, bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.clf()

def plot_residuals(resdict, ncols=4, fs=3, fontsize=15, plotfile='./residual.pdf', 
                   stdscale=5, saveplot=False, showplot=True, ):

    
    # fetch bestfit file
    bffile=os.path.join(resdict['resdir'], resdict['bf_ima'])
    indata=resdict['proc_data']
    
    config=resdict['config_dict']
    fb=config['fitbounds']
    imset=config['ima_set']
    segmap=resdict['seg_data'][fb[2]-1:fb[3],fb[0]-1:fb[1]]
    
    # read input fits file
    ima_hdu=fits.open(bffile)

    hdu_names=[i.name for i in ima_hdu] 
    band_list=[i.name.split('_') [1] for i in ima_hdu  if 'INPUT_' in i.name ]
    
    band_dict={}
    
    for band in band_list:
    
        inp_hdu=ima_hdu[hdu_names.index(f'INPUT_{band}')]
        res_hdu=ima_hdu[hdu_names.index(f'RESIDUAL_{band}')]
        mod_hdu=ima_hdu[hdu_names.index(f'MODEL_{band}')]

        unc_data=imset[band]['unc']
        msk_data=imset[band]['mask']
        
        unc_data=unc_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]
        msk_data=msk_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]
        
        band_dict[band]={'ima_data':inp_hdu.data,
                         'unc_data':unc_data,
                         'res_data':res_hdu.data,
                         'mod_data':mod_hdu.data,
                         'msk_data':msk_data,}

    # plot it
    ndata=len(band_list)
    nrows=int(np.ceil(ndata/ncols))
    nsp=nrows*ncols
    fig, axs = plt.subplots(figsize=(ncols*fs, nrows*fs+0.25), nrows=nrows, ncols=ncols)
    axs=axs.ravel()
    
    for i in range(0, len(band_list)):
        band = band_list[i]
        bdict=band_dict[band]

        ima=bdict['ima_data']
        unc=bdict['unc_data']
        res=bdict['res_data']
        msk=bdict['msk_data']
        #msk=(~bdict['msk_data'].astype(bool)).astype(int)

        nres=res/unc
        nres[msk>0]=np.nan
        #calculate vmin, vam
        std_pix=np.nanstd(nres[segmap==0])
        vmin, vmax=-stdscale*std_pix,stdscale*std_pix
        axs[i].imshow(nres, origin='lower', vmin=vmin, vmax=vmax, cmap='PuOr_r')
        #axs[i].imshow(msk, origin='lower', norm=None, cmap='gray')
        axs[i].set_title(band, fontsize=fontsize)
        axs[i].set_xlim(0, ima.shape[0])
        axs[i].set_ylim(0, ima.shape[1])

    fig.tight_layout()
        
    if saveplot:
        plt.savefig(plotfile, bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.clf()
                      

def plot_psf_subimage(resdict, ncols=4, fs=3, fontsize=15, plotfile='./psf_sub.pdf', 
                      stdscale=5, saveplot=True, showplot=True, save_fits=True):

    
    # fetch bestfit file
    bffile=os.path.join(resdict['resdir'], resdict['bf_ima'])
    scfile=os.path.join(resdict['resdir'], resdict['sc_ima'])
    
    config=resdict['config_dict']
    fb=config['fitbounds']
    imset=config['ima_set']
    segmap=resdict['seg_data'][fb[2]-1:fb[3],fb[0]-1:fb[1]]

    # read input fits file
    ima_hdu=fits.open(bffile)
    scp_hdu=fits.open(scfile)
    hdu_names=[i.name for i in ima_hdu] 
    scp_names=[i.name for i in scp_hdu] 
    band_list=[i.name.split('_') [1] for i in ima_hdu  if 'INPUT_' in i.name ]
    
    band_dict={}
    for band in band_list:
    
        inp_hdu=ima_hdu[hdu_names.index(f'INPUT_{band}')]
        res_hdu=ima_hdu[hdu_names.index(f'RESIDUAL_{band}')]
        mod_hdu=ima_hdu[hdu_names.index(f'MODEL_{band}')]
        inp_wcs=WCS(inp_hdu.header)
        
        unc_data=imset[band]['unc']
        msk_data=imset[band]['mask']
        
        unc_data=unc_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]
        msk_data=msk_data[fb[2]-1:fb[3],fb[0]-1:fb[1]]

        # find PSF model components index
        psf_index=[scp_names.index(i) for i in scp_names if f'_psf_{band}' in i] 

        # loop subtract
        psfsub_data=inp_hdu.data.copy()
        # psf_index=[] for test when there are now PSF models
        for ind in psf_index:
            psfsub_data-=scp_hdu[ind].data

        # save the datasets out
        band_dict[band]={'ima_data':inp_hdu.data, 
                         'psfsub_data': psfsub_data,
                         'unc_data':unc_data,
                         'msk_data':msk_data,
                         'ima_wcs':inp_wcs}


    
    # plot it
    # calcualte normatlization
    refband=resdict['refband']
    refpix=band_dict[refband]['ima_data']
    norm=ImageNormalize(refpix, stretch=LogStretch(), interval=MinMaxInterval())
    
    vmin, vmax=np.nanpercentile(refpix,0.01),np.nanpercentile(refpix,99.95)
    
    ndata=len(band_list)
    nrows=int(np.ceil(ndata/ncols))
    nsp=nrows*ncols
    fig, axs = plt.subplots(figsize=(ncols*fs, nrows*fs+0.25), nrows=nrows, ncols=ncols)
    axs=axs.ravel()

    for i in range(0, len(band_list)):
        band = band_list[i]
        bdict=band_dict[band]
        ima=bdict['ima_data']
        unc=bdict['unc_data']
        msk=bdict['msk_data']
        psfsub=bdict['psfsub_data']
        
        #vmin, vmax=np.nanpercentile(ima,0.05),np.nanpercentile(ima,99.9)
        axs[i].imshow(psfsub, origin='lower', norm=norm, cmap='gray_r')
        #axs[i].imshow(psfsub, origin='lower', vmin=vmin, vmax=vmax, cmap='gray_r')
        #axs[i].imshow(msk, origin='lower', norm=None, cmap='gray')
        axs[i].set_title(band, fontsize=fontsize)
        axs[i].set_xlim(0, ima.shape[0])
        axs[i].set_ylim(0, ima.shape[1])

    fig.tight_layout()
        
    if saveplot:
        plt.savefig(plotfile, bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.clf()

    if save_fits:
        outfits=plotfile.replace('.pdf','.fits')

        pri_hdu=fits.PrimaryHDU(data=segmap, header=None)

        hdulist=[pri_hdu]
        for band in band_list:
            bdict=band_dict[band]
            ima_head=bdict['ima_wcs'].to_header().copy()
            unc_head=bdict['ima_wcs'].to_header().copy()
            msk_head=bdict['ima_wcs'].to_header().copy()

            ima_head['IMATYPE']='IMAGE'
            unc_head['IMATYPE']='ERROR'
            msk_head['IMATYPE']='MASK'
            
            hdulist.append(fits.ImageHDU(data=bdict['ima_data'], header=ima_head))
            hdulist.append(fits.ImageHDU(data=bdict['unc_data'], header=unc_head))
            hdulist.append(fits.ImageHDU(data=bdict['msk_data'], header=msk_head))

        outhdu=fits.HDUList(hdulist)
        outhdu.writeto(outfits, overwrite=True)

def plot_measurements(datadict):
    pass

    
    



        