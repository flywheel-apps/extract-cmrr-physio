
import numpy as np
import utils.physiotools as pt
import matplotlib.pyplot as pl
import os
import pydicom
import os.path as op
import json


class physio:
    def __init__(self ,logfile):

        self.parent_dicom = ''
        self.output_dir = ''
        # Sometimes, in the case of triggers and ECG, there may be multiple channels in the same array.  Thought they
        # Are supposed to be sampled at the same instances in time, sometimes samples are missed, leading to a not
        # quite one to one correspondence of sample times across measures.  We distinguish "raw_values" and "raw_times"
        # As the actual sampled times, and "values" and "acq_times" as any massaged values accounting for (or interpolating)
        # these offsets.
        self.tic_fill_strategy = 'none'
        self.interp_method = 'linear'
        self.fill_val = 0

        self.chan_values = {}        # An array of arrays for ECG and EXT, or a single nested array for RESP and PULS
        self.sig_values = {}
        self.acq_tics = {}         # The acq tics, a final tic time array accounting for missing data.  shared by every array in "values"
        self.acq_times = {}        # same as acq_tics, but in time (seconds)
        self.vol_start = []
        self.proc_qa = {}
        self.proc_max = -1*np.inf
        self.proc_data = False

        self.raw_values = {}     # Raw values pulled right from the .log file
        self.raw_tics = {}       # Raw tic times taken right from the .log file
        self.raw_times = {}
        self.raw_vol_start = []
        self.raw_qa = {}

        self.channels = []
        self.signals = []

        self.tictime = 2.5e-3      # ms per tic
        self.info = []      # Info object with information on the scan

        self.physio_dict = {}      # Dictionary that will be made from the .log file.  Essentially the raw data.
        self.log = False           # Log to write out to
        self.type = ''             # Type of physio log (RESP,PULS,etc).  Taken from the physio_dict object, like most things.
        self.sample_tics = 0       # Sample rate (in tics)
        self.sample_rate = 0       # Sample rate (in Hz)

        if logfile:
            self.logfile = logfile
            self.load_physio()
        else:
            self.logfile = ''


    def load_physio(self):
        try:
            self.physio_dict = pt.log2dict(self.logfile)
            self.sample_tics = int(self.physio_dict['SampleTime'])
            self.sample_rate = 1.0 / (float(self.sample_tics) * float(self.tictime))
            self.raw_values = self.physio_dict['raw_values']
            self.raw_tics = self.physio_dict['raw_tics']
            self.channels = list(self.physio_dict['Chans'])
            self.signals = list(self.physio_dict['Sigs'])
            self.output_dir = os.path.split(self.logfile)[0]

            for chan in self.channels:
                self.raw_times[chan] = self.physio_dict['raw_tics'][chan] * self.tictime

            self.type = self.physio_dict['LogDataType']
        except Exception as e:
            raise Exception("Error loading physio object {}".format(self.logfile)) from e


    def set_info(self,phys_obj):

        try:
            if isinstance(phys_obj, phys_info):
                self.info = phys_obj
            else:
                print('please provide a physio info object')
        except Exception as e:
            raise Exception("Error setting info") from e

    def do_raw_qa(self):

        try:
            for chan in self.channels:
                self.raw_qa[chan] = pt.eval_sampling_quality(self.raw_tics[chan], self.raw_values[chan], self.sample_tics)
        except Exception as e:
            raise Exception("Error doing raw QA for {}".format(self.type)) from e

    def do_proc_qa(self):

        try:
            for chan in self.channels:
                self.proc_qa[chan] = pt.eval_sampling_quality(self.acq_tics[chan], self.chan_values[chan],
                                                             self.sample_tics)
        except Exception as e:
            raise Exception("Error doing processed QA for {}".format(self.type)) from e


    def remove_duplicate_tics(self):

        try:
            for chan in self.channels:
                tics = self.raw_tics[chan]
                values = self.raw_values[chan]
                tics, values = pt.remove_duplicate_tics(tics, values)
                self.acq_tics[chan] = tics
                self.chan_values[chan] = values

        except Exception as e:
            raise Exception("Error removing duplicate values for {}".format(self.type)) from e

    def create_new_tic_array(self):

        #print(self.tic_fill_strategy)

        try:
            min_tic = np.amin(self.physio_dict['ACQ_TIME_TICS'])
            max_tic = np.amax(self.physio_dict['ACQ_TIME_TICS'])
            rate = self.sample_tics

            for chan in self.channels:

                tics = self.acq_tics[chan]

                new_tics = pt.create_new_tics(tics, rate, min_tic, max_tic, option=self.tic_fill_strategy)

                self.acq_tics[chan] = new_tics
                self.acq_times[chan] = self.acq_tics[chan] * self.tictime

            if self.tic_fill_strategy == 'upsample':
                self.sample_tics = 1
                self.sample_rate = 1.0/self.tictime

        except Exception as e:
            raise Exception("Error creating new tic times {}".format(self.type)) from e

    def interp_values_to_newtics(self):

        try:
            if self.interp_method == 'fill':
                fill = self.fill_val
            else:
                fill = self.interp_method

            for chan in self.channels:

                tics = self.raw_tics[chan]
                values = self.raw_values[chan]
                new_tics = self.acq_tics[chan]

                self.chan_values[chan] = pt.interp_vals(tics, values, new_tics, fill=fill)

                mx = max(self.chan_values[chan])

                if mx > self.proc_max:
                    self.proc_max = mx

            self.proc_data = True

        except Exception as e:
            raise Exception("Error interpolating values for {}".format(self.type)) from e

    def triggers_2_timeseries(self):

        try:
            for chan in self.channels:

                self.sig_values[chan] = {}

                for sig in self.signals:
                    self.sig_values[chan][sig] = pt.sig_2_timeseries(self.acq_tics[chan], self.raw_tics[sig])
        except Exception as e:
            raise Exception("Error converting triggers to timeseries for {}".format(self.type)) from e

    def process_raw(self):
        """
        The less efficient but easier to follow method to process data.  Equivalent to "one_step_process_raw", which
        is faster.  Steps:
        1. remove duplicate tics from the raw data
        2. crate a new tic array to handle missing data based on the users "interp" and "handle missing data" settings
        3. interpolate the raw values to the new tic arrays
        4. take trigger signals and convert them to binary timeseries'.

        """
        try:
            self.remove_duplicate_tics()
            #print('Removed Duplicates')
            self.create_new_tic_array()
            #print('Created new Tics')
            self.interp_values_to_newtics()
            #print('Interped Values')
            self.triggers_2_timeseries()
            #print("triggers to ts")
            self.proc_data = True

        except Exception:
            raise

    def minimal_process_raw(self):
        """
        This function is only going to ensure that the channels have the same length by padding at the sample rate
        This is to help BIDS output.
        :return:
        """
        try:
            min_tic = np.amin(self.physio_dict['ACQ_TIME_TICS'])
            max_tic = np.amax(self.physio_dict['ACQ_TIME_TICS'])
            optimal_tics = np.arange(min_tic,max_tic,self.sample_tics)
            optimal_len = len(optimal_tics)
            for chan in self.channels:

                if optimal_len < len(self.raw_values[chan]):
                    optimal_len = len(self.raw_values[chan])

            new_tics = np.arange(0,optimal_len) * self.sample_tics + min_tic
            for chan in self.channels:
                tic_start = np.argmax(new_tics >= self.raw_tics[chan])
                new_vals = np.zeros(optimal_len)
                new_vals[tic_start:] = self.raw_values[chan]
                self.chan_values[chan] = new_vals
                self.acq_tics[chan]=new_tics

                # We need to ensure that all channels have the same length for bids
                self.sig_values[chan] = {}
                for sig in self.signals:
                    self.sig_values[chan][sig] = pt.sig_2_timeseries(self.acq_tics[chan], self.raw_tics[sig])

            self.proc_data = True

        except Exception as e:
            raise Exception("Error performing minimum processing for {}".format(self.type)) from e

    def one_step_process_raw(self):

        try:
            if self.interp_method == 'fill':
                fill = self.fill_val
            else:
                fill = self.interp_method

            self.acq_tics, self.chan_values = pt.process_multichan(self.physio_dict, fill=fill, tic_option=self.tic_fill_strategy)

            for chan in self.channels:

                mx = max(self.chan_values[chan])

                if mx > self.proc_max:
                    self.proc_max = mx

                self.acq_times[chan] = self.acq_tics[chan] * self.tictime
                self.sig_values[chan] = {}

                for sig in self.signals:
                    sig_ts = pt.sig_2_timeseries(self.acq_tics[chan], self.raw_tics[sig])
                    self.sig_values[chan][sig] = sig_ts

            self.proc_data = True
        except Exception as e:
            raise Exception("Error performing one step process for {}".format(self.type)) from e

    def run_full_qc(self):

        try:
            if self.proc_data == False:
                self.minimal_process_raw()

            self.do_raw_qa()
            self.do_proc_qa()
            qa_text = ''
            for chan in self.channels:
                raw_off  = self.raw_qa[chan]['Expected_Time'] - self.raw_qa[chan]['Actual_Time']
                proc_off = self.proc_qa[chan]['Expected_Time'] - self.proc_qa[chan]['Actual_Time']
                raw_max_skip = max(self.raw_qa[chan]['Rate_Count'])
                proc_max_skip = max(self.proc_qa[chan]['Rate_Count'])

                raw_set = set(self.raw_tics[chan])
                proc_set = set(self.acq_tics[chan])
                common = raw_set & proc_set
                pmatch = len(common)/len(raw_set) * 100

                qa_text += '-== {} ==-\n'.format(chan)
                qa_text += 'raw offset: {0:.4f} ms\n'.format(raw_off)
                qa_text += 'raw largest gap: {0:.4f} ms\n'.format(raw_max_skip)
                qa_text += 'proc offset: {0:.4f} ms\n'.format(proc_off)
                qa_text += 'proc largest gap: {0:.4f} ms\n'.format(proc_max_skip)
                qa_text += '% tic match in interp: {0:.2f}\n'.format(pmatch)

            self.plot_physio_qc(qa_text)

        except Exception as e:
            raise Exception("Error running full qc for {}".format(self.type)) from e



    def plot_physio_qc(self,qa_text='',output_path=''):

        try:
            if not output_path:
                output_dir = os.path.split(self.logfile)[0]
                output_path = os.path.join(output_dir,'physio_qa_{}.qa.png'.format(self.type))



            vol_time = self.info.NEW_VOL_TIMES
            acq_start = vol_time[0]
            vol_time = vol_time - acq_start

            # Extract the name of the physiological measure
            phys_name = self.type

            # Generate a subplot and set up some aesthetics
            fig = pl.figure(figsize=(12, 5))

            if qa_text:
                ax = fig.add_axes((.25, .1, .75, .9))
            else:
                ax = fig.add_axes((.1, .1, .85, .9))

            ax.grid(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Plot the physio data


            for ci, chan in enumerate(self.channels):

                # If it's the EXT trigger channel, we ONLY plot the signals
                if not phys_name == 'EXT':
                    ax.plot(self.acq_times[chan]-acq_start, self.chan_values[chan] + self.proc_max * ci,linewidth=0.7,label='physio {}'.format(chan))


            for si, sig in enumerate(self.signals):
                ax.plot(self.acq_times[chan]-acq_start, (self.sig_values[chan][sig]) * (self.proc_max / 10) - (self.proc_max / 10)*si*1.5, linewidth=0.7, label='{} {}'.format(chan, sig))


            # Get the natural y axis limits and plot two vertical lines, indicating the start and stop of the scan
            yl, yh = ax.get_ylim()
            #print(yl)
            #print(yh)
            ax.plot([vol_time[0], vol_time[0]], [yl, yh], color='g', linewidth=1.5, label='Volume Acquisition start')
           # print(phys_name)
            #print(vol_time[0])

            # Since our new_vol measurements are the time of the start of a volume acquisition, the true end of the scan
            # is one TR after the last acquisition start.  So, we can just calculate a "dt" and modify our end time
            dt = np.mean(np.diff(vol_time))
            ax.plot([vol_time[-1] + dt, vol_time[-1] + dt], [yl, yh], color='r', linewidth=1.5, label='Volume Acquisition end')
            #print(vol_time[-1])
            #print(dt)

            # Set the plot layout to tight and add a legend
            ax.set_title(phys_name)
            #pl.tight_layout()
            if qa_text:
                pl.text(-.32, -.08, qa_text, family='serif',fontsize=8, ha='left', transform=ax.transAxes)
            pl.legend(loc='best')

            # Save the figure and close
            pl.savefig(output_path, dpi=250)
            pl.close()
        except Exception as e:
            raise Exception("Error plotting qc for {}".format(self.type)) from e


    def plot_raw(self):

        pl.figure()
        for chan in self.channels:
            pl.plot(self.raw_tics[chan],self.raw_values[chan])

    def plot_proc(self):
        pl.figure()
        for chan in self.channels:
            pl.plot(self.acq_tics[chan],self.chan_values[chan])




    def bids_o_matic_9000(self, processed=True, matches='', zip_output=False):
        """
        This function takes physio.log data that's been convered to physio dict objects (stored in gear context custom_dict{}
        This creates files from those dictionaries in BIDS format.

        :param raw_dicom: the raw dicom file (incase we need to go into it for naming purposes)
        :param context: the gear context
        :return: NOTHING
        """
        try:
            if processed == False:
                print('Note: its highly recommended that the data is processed before bidsifying')
                # Do minimal processing on raw (align tics and create trigger timeseries')
                self.minimal_process_raw()


            json_dict = {}
            output_dir = self.output_dir
            # Grab useful information from the input object:
            # variable names are following bids tutorial:
            # https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continous-recordings.html
            #context.log.debug('extracting matches value')
            #context.log.debug(print(context.get_input('DICOM_ARCHIVE')))
            #context.log.debug(print(raw_dicom))

            #(chan_values, sig_values, new_vol_ticks)
            # Assuming that all the channels now share the same timeseries, only need one chans' sig tics.  I hope.
            temp_chan = self.channels[0]
            temp_tics = self.acq_tics[temp_chan]

            # Extract the new_vol_tics from the info dict
            info_dict = self.info
            new_vol_ticks, new_vol_array = info_dict.match_volstart_to_tic(temp_tics)

            bids_file, json_header = pt.dicts2bids(self.chan_values, self.sig_values[temp_chan], new_vol_array)

            # Find the sample interval (in tics)
            json_dict['SamplingFrequency'] = self.sample_rate

            if not matches:

                # If matches isn't provided, we'll load the dicom and find something from there...
                # But first warn the user
                # Load the original dicom for naming info
                raw_dicom = self.parent_dicom
                dicom = pydicom.read_file(raw_dicom)


                # If the protocol name is empty in the dicom, then this is all just messed up
                if not dicom.ProtocolName:
                    # context.log.warning('Dicom header is missing values.  Unable to properly name BIDS physio file')
                    # context.log.warning('User will need to manually rename')

                    # Let the user manually assign the name later
                    matches = 'UnknownAcquisition'
                else:
                    # Otherwise use what's provided in the dicom's Protocol name
                    # context.log.debug('Settin matchis in the dicom protocol else loop')
                    matches = '{}'.format(dicom.ProtocolName)


            # Now create and save the .json file
            header_lookup = {'EXT':'triggers', 'RESP':'respiration', 'PULS':'pulse', 'ECG':''}
            hdr = header_lookup[self.type]

            if hdr:
                inds = np.where([i == self.type for i in json_header])[0][0]
                json_header[inds] = hdr
                label = hdr
            else:
                label = self.type

            # Set the file name and save the file
            physio_output = op.join(output_dir, '{}_recording-{}_physio.tsv'.format(matches, label))
            np.savetxt(physio_output, bids_file, fmt='%d', delimiter='\t')

            if zip_output:
                # Zip the file
                gzip_command = ['gzip', '-f', physio_output]
                # exec_command(context, gzip_command)
                # context.custom_dict['physio-dicts'][physio]['bids_tsv'] = physio_output + '.gz'



            json_output = op.join(output_dir, '{}_recording-{}_physio.json'.format(matches, label))
            json_dict['StartTime'] = (temp_tics[0] - info_dict.ACQ_START_TICS[0]) * self.tictime
            json_dict['Columns'] = json_header

            with open(json_output, 'w') as json_file:
                json.dump(json_dict, json_file)

            # EASY PEASY LIVIN GREEZY

        except Exception as e:
            raise Exception("Error generating BIDS files for {}".format(self.type)) from e
        pass








class phys_info:
    def __init__(self, logfile):
        self.info_dict = {}
        self.tic_len = 2.5e-3
        self.ACQ_START_TICS = []
        self.NEW_VOL_TICS = []
        self.NEW_VOL_TIMES = []

        self.info_dict = pt.log2dict(logfile)

        self.NEW_VOL_TICS, self.NEW_VOL_TIMES = self.get_volume_start_tics()

    def get_volume_start_tics(self):
        try:
            # Let's convert the "volume" and "ticks" array to int
            self.info_dict['VOLUME'] = np.array(self.info_dict['VOLUME']).astype(int)
            self.info_dict['ACQ_START_TICS'] = np.array(self.info_dict['ACQ_START_TICS']).astype(int)

            # Take the difference to see when it changes and extract the ticks at those changes
            new_vol_inds = np.where(np.diff(self.info_dict['VOLUME']))[0] + 1
            new_vol_ticks = self.info_dict['ACQ_START_TICS'][new_vol_inds]
            self.info_dict['NEW_VOL_TICS'] = new_vol_ticks
            self.info_dict['NEW_VOL_TIMES'] = new_vol_ticks * self.tic_len

            self.ACQ_START_TICS = np.array(self.info_dict['ACQ_START_TICS']).astype(int)

            return new_vol_ticks, new_vol_ticks * self.tic_len

        except Exception as e:
            raise Exception("Error getting volume start tics for {}".format(self.logfile)) from e


    def match_volstart_to_tic(self, tics):
        try:
            last_tic_vol = -1
            new_tics = []
            new_vol_array = np.zeros(len(tics))

            for i, tic in enumerate(self.NEW_VOL_TICS):

                current_tic_vol = np.argmax(tics >= tic)

                if current_tic_vol > last_tic_vol:
                    new_vol_array[current_tic_vol] = 1
                    new_tics.append(current_tic_vol)

                    last_tic_vol = current_tic_vol

            return new_tics, new_vol_array

        except Exception as e:
            raise Exception("Error matching volume start tics for {}".format(self.logfile)) from e








# ext = '/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/TestPhysioData/cmrr_mbep2d_Physio_PhysioLog/Physio_20190515_143710_a231775f-6990-4793-881e-3d55bb283f9b_EXT.log'
# ext_info='/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/TestPhysioData/cmrr_mbep2d_Physio_PhysioLog/Physio_20190515_143710_a231775f-6990-4793-881e-3d55bb283f9b_Info.log'
#
#
# info = phys_info('/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/extract-cmrr-physio/output/func-bold_task-lokicat_run-01_Info.log')
# puls = '/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/extract-cmrr-physio/output/func-bold_task-lokicat_run-01_PULS.log'
# resp = '/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/extract-cmrr-physio/output/func-bold_task-lokicat_run-01_RESP.log'
# ecg = '/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/extract-cmrr-physio/output/func-bold_task-lokicat_run-01_ECG.log'
#
#
# qnlresp ='/Users/davidparker/Documents/Flywheel/SSE/MyWork/Gears/BIDs_Physio/TestPhysioData/Physio_20180730_190934_83cd52f7-d767-4959-b441-210aeab19fe3_RESP.log'
# test = physio(resp)
#
#
#
# test.set_info(info)
# test.fill_val = 0
# test.interp = 'fill'
# test.tic_fill_strategy = 'upsample'
# test.process_raw()
# test.run_full_qc()
# test.bids_o_matic_9000(processed=True, matches='my_match', zip_output=False)
#

