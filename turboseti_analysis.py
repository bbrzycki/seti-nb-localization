import sys
import numpy as np
import os
import time

from astropy import units as u
import blimpy as bl
import setigen as stg
from turbo_seti.find_doppler.find_doppler import FindDoppler


def rmse(true, pred):
    # Calculates RMSE in index units
    return np.mean((true - pred)**2)**0.5


def main():
    experiment_path = '/datax/scratch/bbrzycki/data/nb-localization/training/'
    output_dir = experiment_path + 'turboseti/'

    turbo_rmse_dict = {}
    wrong_num_signals = 0
    timing_array = []

    turbo_start = time.time()
    for db in range(0, 30, 5):
        for j in range(4000):
            print(db, j)

            fn = '{:02}db_{:06d}.npy'.format(db, j)
            npy_fn = '/datax/scratch/bbrzycki/data/nb-localization/1sig/test/{}'.format(fn)
            fil_fn = output_dir + '{:02}db_{:06d}.fil'.format(db, j)
            dat_fn = output_dir + '{:02}db_{:06d}.dat'.format(db, j)

            frame = stg.Frame(fchans=1024,
                              tchans=32,
                              df=1.3969838619232178*u.Hz,
                              dt=1.4316557653333333*u.s,
                              fch1=6095.214842353016*u.MHz,
                              data=np.load(npy_fn))
            frame.save_fil(fil_fn)

            try:
                os.remove(dat_fn)
            except FileNotFoundError:
                pass

            start_time = time.time()

            find_seti_event = FindDoppler(fil_fn,
                                          max_drift=31,
                                          snr=10,
                                          out_dir=output_dir)
            find_seti_event.search()

            end_time = time.time()
            timing_array.append(end_time - start_time)

            with open(dat_fn, 'r') as f:
                data = [line.split() for line in f.readlines() if line[0] != '#']

            # Count number of times turboseti predicts wrong number
            if len(data) != 1:
                wrong_num_signals += 1

            estimates = []
            snrs = []
            for signal in data:
                snr = float(signal[2])
                drift_rate = float(signal[1])
                start_index = 1024 - int(signal[5])
                end_index = frame.get_index(frame.get_frequency(start_index) + drift_rate * frame.tchans * frame.dt)

                estimates.append([start_index, end_index])
                snrs.append(snr)
                
            if len(estimates) != 0:
                # Get the ground truth positions from the saved dictionaries
                true_indices = (np.load(experiment_path + 'final_1sig_32bs_bright/test_predictions.npy',
                                            allow_pickle=True).item()[fn] * 1024)[0]

                # If turboseti found signals, choose the highest SNR one
                turbo_rmse_dict[fn] = rmse(true_indices, estimates[np.argsort(snrs)[-1]])

    timing_array = np.array(timing_array)
    print('Wrong: {} frames'.format(wrong_num_signals))
    print('Total search: {:.2f} seconds'.format(time.time() - turbo_start))
    
    np.save(output_dir + 'timing_array.npy', timing_array)
    np.save(output_dir + 'test_predictions.npy', turbo_rmse_dict)



if __name__ == '__main__':
    main()





