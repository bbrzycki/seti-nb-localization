import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os, glob, errno
import csv
import json
import time

import numpy as np
import pandas as pd
import blimpy as bl
from astropy import units as u

sys.path.insert(0, "../setigen/")
import setigen as stg


def db_to_snr(db):
    return np.power(10, db / 10)


def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def generate_frame(sig_db, rfi_num=0):
    """
    Create frame with chi-squared synthetic noise, with 1 drifting signal
    and either 0 or 1 non-drifting 'RFI' signals.
    """
    frame = stg.Frame(fchans=1024,
                      tchans=32,
                      df=1.3969838619232178*u.Hz,
                      dt=1.4316557653333333*u.s,
                      fch1=6095.214842353016*u.MHz)
    
    frame.add_noise_from_obs()
    noise_mean, noise_std = frame.get_noise_stats()

    start_index = np.random.randint(0, frame.fchans)
    end_index = np.random.randint(0, frame.fchans)
    drift_rate = frame.get_drift_rate(start_index, end_index)

    width = np.random.uniform(1, 30)

    frame.add_constant_signal(f_start=frame.get_frequency(start_index),
                              drift_rate=drift_rate*u.Hz/u.s,
                              level=frame.get_intensity(snr=db_to_snr(sig_db)),
                              width=width*u.Hz,
                              f_profile_type='gaussian')

    if rfi_num == 1:
        rfi_snr = db_to_snr(25)
        rfi_start_index = rfi_end_index = np.random.randint(0, frame.fchans)
        rfi_width = np.random.uniform(1, 30)
        
        frame.add_constant_signal(f_start=frame.get_frequency(rfi_start_index),
                                  drift_rate=0*u.Hz/u.s,
                                  level=frame.get_intensity(snr=rfi_snr),
                                  width=rfi_width*u.Hz,
                                  f_profile_type='gaussian')
    else:
        rfi_start_index = rfi_end_index = -1
    
    frame_info = {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'sig_db': sig_db,
        'start_index': start_index,
        'end_index': end_index,
        'width': width,
        'rfi_num': rfi_num,
        'rfi_start_index': rfi_start_index,
        'rfi_end_index': rfi_end_index,
    }
    
    return frame, frame_info


if __name__ == '__main__':
    start_time = time.time()
    path = '/datax/scratch/bbrzycki/data/nb-localization'

    # Numbers for each subcategory; # rfi (0,1), # signals (1), db (0,5,10,15,20,25)
    splits = [('train', 20000), ('test', 4000)]
    
    for rfi_num in range(2):
        xsig = '{:d}sig'.format(1 + rfi_num)
        
        for i, split in enumerate(splits):
            # First, create directories for data
            split_name, split_num = split
            mkdir('{}/{}/{}'.format(path, xsig, split_name))

            info_list = []
            for sig_db in [0, 5, 10, 15, 20, 25]:
                for j in range(split_num):
                    fn = '{:02}db_{:06d}.npy'.format(sig_db, j)

                    # Generate signals
                    frame, frame_info = generate_frame(sig_db, rfi_num=rfi_num)
                    frame.save_npy('{}/{}/{}/{}'.format(path,
                                                        xsig,
                                                        split_name,
                                                        fn))

                    # Directly link filename to the frame info for convenience
                    frame_info['filename'] = fn
                    info_list.append(frame_info)

                    print('Saved frame {} for db {:d} {} set, with {:d} rfi signals'.format(j, sig_db, split_name, rfi_num))
                    
            # Save out labels using pandas
            df = pd.DataFrame(info_list)
            df.to_csv('{}/{}/{}/labels.csv'.format(path, 
                                                   xsig,
                                                   split_name), index=False)

    end_time = time.time()
    print('Dataset generation finished in {:.2f} minutes'.format((end_time - start_time)/60))