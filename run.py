#!/usr/bin/env python3

import flywheel
from utils import futils as fu
import logging
from utils.Common import exec_command
import os.path as op
import os
import glob
import sys
import numpy as np
import json
import pydicom
import shutil
import matplotlib.pyplot as pl
import pickle

##-------- Standard Flywheel Gear Structure --------##
flywheelv0 = "/flywheel/v0"
environ_json = '/tmp/gear_environ.json'

##--------  Gear Specific files/folders/values  ----##
tic_len = 2.5e-3  # seconds, length of one "tick"


def physio_qc(physio_dict,new_vol_tics,output_path):

    """
    :param physio_dict: a dictionary object made from a physio.log file
    :param new_vol_tics: an array of the tic times for new volumes during the acquistion
    :param output_dir:  the output directory to save the resulting images to
    :return: nothing

    This function generates a visual plot of the physiological data for quick inspection
    """

    # First covert tics to seconds for the physio and new volume tic times
    physio_time = np.array(physio_dict['ACQ_TIME_TICS']).astype(float) * tic_len
    vol_time = new_vol_tics.astype(float) * tic_len

    # Now we'll recenter the time so time t=0 is the beginning of the scan
    acq_start = vol_time[0]
    vol_time = vol_time - acq_start
    physio_time = physio_time - acq_start

    # Extract the name of the physiological measure
    phys_name = physio_dict['LogDataType']

    # Now ensure that the physio values are a plottable format
    # May need to modify this for ECG
    if phys_name == 'EXT':
        triggers = physio_dict['Triggers']
        physio_vals = []
        names = []
        for trigger in triggers:
            physio_vals.append(np.array(physio_dict[str(trigger)]))
            names.append(str(trigger))

    else:
        physio_vals = [np.array(physio_dict['VALUE']).astype(int)]
        names = [phys_name]

    # Generate a subplot and set up some aesthetics
    f, ax = pl.subplots(1, 1, figsize=(10, 4))
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot the physio data
    for val, name in zip(physio_vals,names):
        ax.plot(physio_time, val, linewidth=0.7, label='physio {}'.format(name))

    # Get the natural y axis limits and plot two vertical lines, indicating the start and stop of the scan
    yl, yh = ax.get_ylim()
    ax.axvline(vol_time[0],yl,yh,color='g',linewidth=1.5,label='Volume Acquisition start')

    # Since our new_vol measurements are the time of the start of a volume acquisition, the true end of the scan
    # is one TR after the last acquisition start.  So, we can just calculate a "dt" and modify our end time
    dt = np.mean(np.diff(vol_time))
    ax.axvline(vol_time[-1]+dt,yl,yh,color='r',linewidth=1.5,label='Volume Acquisition end')

    # Set the plot layout to tight and add a legend
    ax.set_title(phys_name)
    pl.tight_layout()
    pl.legend()

    # Save the figure and close
    pl.savefig(output_path, dpi=250)
    pl.close()

    pass


def run_qc_on_physio_logs(context):
    """
    This function runs the function to generate quality control measures (graphs) of the physio output
    :param context:
    :return:
    """

    # Expected physio type outputs from conversion tool
    expected_physio = ['info','PULS','ECG','PULS','EXT']

    # First extract the info, we need new_vol_tics
    info = context.custom_dict['physio-dicts']['info']
    new_vol_tics = info['NEW_VOL_TICS']

    for physio in context.custom_dict['physio-dicts']:

        # If this dict is info, skip
        if physio == 'info':
            continue

        # Otherwise extract the dict and pass that to the qc function
        physio_dict = context.custom_dict['physio-dicts'][physio]
        output_path = op.join(context.output_dir,'{0}_{1}.qa.png'.format(context.custom_dict['matches'], physio_dict['LogDataType']))
        physio_qc(physio_dict, new_vol_tics, output_path)

    return

def log2dict(physio_log,context=''):
    """
    This function takes a standard physio log file as output by CMRR-MB (https://github.com/CMRR-C2P/MB/)
    and converts them to machine-friendly python dictionaries
    :param physio_log: the .log file produced by the CMRR-MB script
    :param context: the gear context
    :return: the dictionary is returned
    """



    physio_dict = {}
    header = []
    # context.log.debug('loading file')
    f = open(physio_log, 'r')
    # context.log.debug('going through lines')
    for line in f.readlines():

        # Remove trailing endline
        line = line.strip('\n')

        # If there's an equals sign, we're still in the header, and we need to store this value
        if line.find('=') > -1:

            # Create a key, value pair by splitting the line by the equals sign
            pair = line.split('=')

            # store the key and value in the dict (everything as strings for now)
            # After stripping whitespace of course
            physio_dict[pair[0].strip()] = pair[1].strip()

        # Otherwise If it's an empty line, skip it
        elif line == '':
            continue

        # Otherwise, if we remove the spaces and underscores and it's alpha, then it's headers and we need it
        elif line.replace(' ', '').replace('_','').isalpha():
            header = line.split()
            for h in header:
                physio_dict[h] = []

        # if the first values is numeric (the tic value), it's data BUT
        elif line.split()[0].isnumeric():

            # if headers are empty, we hit numbers before we hit headers
            if not header:
                raise Exception('Unable to parse .log files, no headers found')

            vals = line.split()

            # If there are missing values, such as a trigger pulse,
            # We append values to the end.  Yes, we are assuming they're at the end
            # Given the converter they're using, they seem to try to use tabs to separate columns, but
            # Those apparently end up converted to spaces, so there's no real way of knowing how much whitespace
            # there is between values.  Comma separated would be helpful in the future.
            while len(vals) < len(header):
                vals.append('')

            for h, v in zip(header, vals):
                physio_dict[h].append(v)

        else:
            print('Unrecognized line: {}\nskipping'.format(line))

    # Now we will perform some special modifications depending on the file type:
    # At the moment we only have special tasks for "EXT"
    if physio_dict['LogDataType'] == 'EXT':
        physio_array = physio_dict['SIGNAL']
        ext_triggers = set(physio_array)
        ext_triggers.discard('')
        physio_dict['Triggers'] = ext_triggers

        for trigger in ext_triggers:

            # Make separate arrays for triggers in the column.  I think this is how it works...
            trigger_array = [int(i == trigger) for i in physio_array]
            physio_dict[str(trigger)] = trigger_array


    elif physio_dict['LogDataType'] == 'ECG':

        # Hi future flywheel.  It's me, David, from the year 2019.  I honestly just don't have any example ECG data so
        # I don't know what the log files look like.  Sorry.  If you encounter someone with this data, please work with
        # them to develop this feature.  This code flows as follows: matlab code makes .log files, this function reads
        # them and makes machine-friendly python dictionaries, and then those are converted to BIDS compliant data.
        # I just wanted to tell you both good luck, we're all counting on you.

        raise Exception("ECG is unsupported.  Please see Flywheel, and they will work with you to develop this feature")


    return physio_dict



def create_physio_dicts_from_logs(context):

    """
    This function generates easily accessible dictionary objects for each physio log file

    :param context: The gear context
    :return:
    """

    # Set the output dir
    physio_output_dir = context.output_dir

    # First let's check the output.  We'll try to make this robust for any measurement:
    context.log.debug('Globbing Physio')
    all_physio = glob.glob(op.join(physio_output_dir, '*.log'))
    info = glob.glob(op.join(physio_output_dir, '*Info.log'))[0]

    # Handle errors and warnings about missing physio data
    if len(all_physio) == 0 or (len(all_physio) == 1 and len(info) == 1):
        context.log.warning('No physio signals extracted')
        return


    # First we need to load the acquisition info file, and look through to see when the volumes were acquired
    # I believe we need this for the "trigger".  Typically, the scanner triggers at every volume.  I'm ASSUMING
    # That's what they're looking for, but it's not completely clear.
    context.log.debug('trying log2dict for info')
    info_dict = log2dict(info, context)

    # Let's convert the "volume" and "ticks" array to int
    info_dict['VOLUME'] = np.array(info_dict['VOLUME']).astype(int)
    info_dict['ACQ_START_TICS'] = np.array(info_dict['ACQ_START_TICS']).astype(int)

    # Take the difference to see when it changes and extract the ticks at those changes
    new_vol_inds = np.where(np.diff(info_dict['VOLUME']))[0] + 1
    new_vol_ticks = info_dict['ACQ_START_TICS'][new_vol_inds]
    info_dict['NEW_VOL_TICS'] = new_vol_ticks

    # Store this as dictionary in our context custom dict.
    context.custom_dict['physio-dicts']['info'] = info_dict



    # See if the "info" tag is filled and has the "Series Description" key:
    matches = context.get_input('DICOM_ARCHIVE')['object']['info']

    if 'SeriesDescription' in matches.keys() and matches['SeriesDescription']:
        # Use that
        matches = matches['SeriesDescription']
    else:
        # If not, we'll load the dicom and find something from there...
        # But first warn the user
        # Load the original dicom for naming info
        raw_dicom = context.custom_dict['raw_dicom']
        dicom = pydicom.read_file(raw_dicom)
        context.log.warning('dicom metadata missing values.  Extracting "protocol name" from dicom for BIDS naming')
        context.log.debug('loading {}'.format(raw_dicom))
        context.log.debug('dicom protocol name: {}'.format(dicom.ProtocolName))

        # If the protocol name is empty in the dicom, then this is all just messed up
        if not dicom.ProtocolName:
            context.log.warning('Dicom header is missing values.  Unable to properly name BIDS physio file')
            context.log.warning('User will need to manually rename')

            # Let the user manually assign the name later
            matches = 'UnknownAcquisition'
        else:
            # Otherwise use what's provided in the dicom's Protocol name
            context.log.debug('Settin matchis in the dicom protocol else loop')
            matches = '{}'.format(dicom.ProtocolName)

    context.custom_dict['matches'] = matches

    context.log.info('matches value:')
    context.log.info('matches: {}'.format(matches))

    context.log.debug('matches type: {}'.format(type(matches)))

    for physio in all_physio:

        # If this in the info file, we don't do it.
        if physio == info:
            new_file = op.join(context.output_dir,'{}_Info.log'.format(context.custom_dict['matches']))
            shutil.move(physio, new_file)
            continue

        # Otherwise make it a dictionary:
        physio_dict = log2dict(physio, context)
        context.log.debug('complete')
        physio_dict['ACQ_TIME_TICS'] = np.array(physio_dict['ACQ_TIME_TICS']).astype(int)
        physio_dict['VALUE'] = np.array(physio_dict['VALUE']).astype(int)

        # Store this as a dictionary in our context custom dict
        context.custom_dict['physio-dicts'][physio_dict['LogDataType']] = physio_dict

        # And rename the log file to something sane:
        new_file = op.join(context.output_dir, '{0}_{1}.log'.format(context.custom_dict['matches'], physio_dict['LogDataType']))
        shutil.move(physio, new_file)

    # The dictionaries are now stored in context.  We can return to the calling function
    return



def dicts2bids(physio_dict, new_vol_ticks):
    """
    This function takes the physio dictionaries generated by create_physio_dicts_from_logs() and converts them to BIDS
    compliant files
    :param physio_dict: The dictionaries that contain the physiological recordings and info about the acquisition
    :param new_vol_ticks: start times of each volume in the acquisition
    :return: the bids file, a nx2 matrix, where n is the number of physio sample points, column 0 is the physio
    recorxing, and column 1 is the scanner trigger column.
    """

    # Create an empty BIDS file.  this has the physio value and the scan trigger
    phys_tics = physio_dict['ACQ_TIME_TICS']


    # Expected physio type outputs from conversion tool
    # expected_data_type = ['ACQUISITION_INFO', 'PULS', 'ECG', 'PULS', 'EXT']
    # Note: This is based off one set of sample data that is no means exhaustive.
    # It is likely that this code does not match all use-cases in other data.

    if physio_dict['LogDataType'] == 'EXT':
        # For debugging
        # f = open('/flywheel/v0/output/physiodict.pkl','wb')
        # pickle.dump(physio_dict,f)
        # f.close()

        ext_triggers = physio_dict['Triggers']
        phys_valu = physio_dict['VALUE']
        bids_file = np.zeros((len(phys_valu), len(ext_triggers)+1))
        vol = 0

        for trigger in ext_triggers:

            trigger_array = physio_dict[str(trigger)]

            for i, tic in enumerate(phys_tics):
                bids_file[i, 0] = trigger_array[i]
                # If we're not at the next to last volume
                if vol < len(new_vol_ticks) - 1:

                    # Check if the current tic is greater than or equal to the next volume tic,
                    # Then this is as close as we can get to a scanner trigger, so set it as 1
                    next_vol_tick = new_vol_ticks[vol + 1]
                    if tic >= next_vol_tick:
                        bids_file[i, 1] = 1

                        # and increment the volume
                        vol += 1

    else:

        phys_valu = physio_dict['VALUE']
        bids_file = np.zeros((len(phys_valu), 2))
        vol = 0

        for i, tic in enumerate(phys_tics):

            bids_file[i, 0] = phys_valu[i]

            # If we're not at the next to last volume
            if vol < len(new_vol_ticks) - 1:

                # Check if the current tic is greater than or equal to the next volume tic,
                # Then this is as close as we can get to a scanner trigger, so set it as 1
                next_vol_tick = new_vol_ticks[vol + 1]
                if tic >= next_vol_tick:
                    bids_file[i, 1] = 1

                    # and increment the volume
                    vol += 1

    return bids_file


def bids_o_matic_9000(raw_dicom, context):
    """
    This function takes physio.log data that's been convered to physio dict objects (stored in gear context custom_dict{}
    This creates files from those dictionaries in BIDS format.

    :param raw_dicom: the raw dicom file (incase we need to go into it for naming purposes)
    :param context: the gear context
    :return: NOTHING
    """

    # Set the output dir
    physio_output_dir = context.output_dir

    # First let's check the output.  We'll try to make this robust for any measurement:
    context.log.debug('Globbing Physio')
    all_physio = glob.glob(op.join(physio_output_dir, '*.log'))
    info = glob.glob(op.join(physio_output_dir, '*Info.log'))[0]
    json_dict = {}

    # Grab useful information from the input object:
    # variable names are following bids tutorial:
    # https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continous-recordings.html
    context.log.debug('extracting matches value')
    context.log.debug(print(context.get_input('DICOM_ARCHIVE')))
    context.log.debug(print(raw_dicom))




    # Handle errors and warnings about missing physio data
    if len(all_physio) == 0 or (len(all_physio) == 1 and len(info) == 1):
        context.log.warning('No physio signals extracted')
        return


    # Extract the new_vol_tics from the info dict
    info_dict = context.custom_dict['physio-dicts']['info']
    new_vol_ticks = info_dict['NEW_VOL_TICS']

    for physio in context.custom_dict['physio-dicts']:

        # If this key is the info key, skip it
        if physio == 'info':
            continue

        # Extract the dictionary
        physio_dict=context.custom_dict['physio-dicts'][physio]

        # Make a bids compatible array
        context.log.debug('running bids2dict')
        bids_file = dicts2bids(physio_dict, new_vol_ticks)
        context.log.debug('success')

        # If we can see that it's RESP or PULS, give it a nice name.
        # Otherwise we'll keep whatever's in the header.
        label = physio_dict['LogDataType']
        # if label == "RESP":
        #     label = 'respiration'
        # elif label == "PULS":
        #     label = 'cardiac'

        # Find the sample interval (in tics)
        json_dict['SamplingFrequency'] = 1.0 / float(physio_dict['SampleTime']) * tic_len

        matches = context.custom_dict['matches']

        # Set the file name and save the file
        physio_output = op.join(context.output_dir,'{}_recording-{}_physio.tsv'.format(matches, label))
        np.savetxt(physio_output, bids_file, fmt='%d', delimiter='\t')

        # Zip the file
        gzip_command = ['gzip','-f',physio_output]
        exec_command(context, gzip_command)

        # Now create and save the .json file
        if physio == 'EXT':
            json_output = op.join(context.output_dir,'{}_recording-{}_physio.json'.format(matches,label))
            json_dict['StartTime'] = (physio_dict['ACQ_TIME_TICS'][0] - info_dict['ACQ_START_TICS'][0]) * tic_len
            label = list(physio_dict['Triggers'])
            label.append('trigger')
            json_dict['Columns'] = label

        # May need to add a case here for ECG

        else:
            json_output = op.join(context.output_dir,'{}_recording-{}_physio.json'.format(matches,label))
            json_dict['StartTime'] = (physio_dict['ACQ_TIME_TICS'][0] - info_dict['ACQ_START_TICS'][0]) * tic_len
            json_dict['Columns'] = [label,'trigger']


        with open(json_output,'w') as json_file:
            json.dump(json_dict, json_file)


        # EASY PEASY LIVIN GREEZY

    pass




def main():
    #shutil.copy('config.json','/flywheel/v0/output/config.json')

    supported_filetypes = [".dcm",".IMA"]


    with flywheel.gear_context.GearContext() as gear_context:

        #### Setup logging as per SSE best practices
        fmt = '%(asctime)s %(levelname)8s %(name)-8s %(funcName)s - %(message)s'
        logging.basicConfig(level=gear_context.config['gear-log-level'], format=fmt)
        gear_context.log = logging.getLogger('[flywheel/extract-cmrr-physio]')
        gear_context.log.info('log level is ' + gear_context.config['gear-log-level'])
        gear_context.log_config()  # not configuring the log but logging the config

        # Set up Custom Dicionary to host user variables
        gear_context.custom_dict = {}

        gear_context.custom_dict['environ'] = environ_json
        # Create a 'dry run' flag for debugging
        gear_context.custom_dict['dry-run'] = gear_context.config['Dry-Run']

        # Set up a field for physio dictionaries (used for bidsifying and qc)
        gear_context.custom_dict['physio-dicts'] = {}

        # Now let's set up our environment from the .json file stored in the docker image:
        environ = fu.set_environment(environ_json, gear_context.log)
        output_dir = gear_context.output_dir

        # Now we need to extract our input file, and check if it exists
        dicom = gear_context.get_input_path('DICOM_ARCHIVE')
        run_bids = gear_context.config['Generate_Bids']

        fu.exists(dicom, gear_context.log, '.zip')
        zip_base = op.splitext(op.split(dicom)[-1])[0]

        # Now we need to unzip it:
        uz_dir = op.join('/tmp', 'unzipped_dicom')
        unzip_dicom_command = ['unzip','-o', dicom, '-d', uz_dir]

        ###########################################################################
        # Try to run the unzip command:
        try:
            exec_command(gear_context, unzip_dicom_command)
        except Exception as e:
            gear_context.log.fatal(e, )
            gear_context.log.fatal('The CMRR physio extraction Failed.', )
            os.sys.exit(1)

        # Now we locate the raw unzipped dicom
        for ft in supported_filetypes:
            raw_dicom = glob.glob(op.join(uz_dir, zip_base, '*{}'.format(ft)))
            gear_context.log.info('Looking for {}'.format(op.join(uz_dir, zip_base, '*{}'.format(ft))))

            # If we found an expected filetype, let's assume that's what were looking for
            if len(raw_dicom) == 1:
                break

        # If we found too many, exit
        if len(raw_dicom) > 1:
            print(raw_dicom)
            gear_context.log.fatal(
                'Dicom structure contains too many dicoms, unrecognized CMRR physio format for this gear')

            sys.exit(1)

        # IF we didn't find any, exit
        elif len(raw_dicom) < 1:
            gear_context.log.fatal('Dicom structure unzipped zero file of supported file type.')
            sys.exit(1)


        run_physio_command = ['/usr/local/bin/extractCMRRPhysio', raw_dicom[0], output_dir]
        gear_context.custom_dict['raw_dicom'] = raw_dicom[0]

        ###########################################################################
        # Try to run the extract physio command:
        try:
            exec_command(gear_context, run_physio_command)
        except Exception as e:
            gear_context.log.fatal(e, )
            gear_context.log.fatal('The CMRR physio extraction Failed.', )
            os.sys.exit(1)

        gear_context.log.debug('Successfully extracted physio')

        ###########################################################################
        # Try to generate physio dictionaries from logs (used for QC and BIDS)
        try:
            create_physio_dicts_from_logs(gear_context)
        except Exception as e:
            gear_context.log.fatal(e, )
            gear_context.log.fatal('Unable to create Physio Dictionaries', )
            os.sys.exit(1)

        gear_context.log.debug('Successfully generated physio dict')


        ###########################################################################
        # Try to run physio QC command:
        # Make QC file:

        gear_context.log.debug('Performing PhysioQC')
        try:
            run_qc_on_physio_logs(gear_context)
        except Exception as e:
            gear_context.log.warning(e, )
            gear_context.log.warning('Unable to run qc', )

        ###########################################################################
        # Try to run the bidsify command:

        if run_bids:

            try:
                bids_o_matic_9000(raw_dicom[0],gear_context)
            except Exception as e:
                gear_context.log.fatal(e, )
                gear_context.log.fatal('The CMRR conversion to BIDS failed', )

                os.sys.exit(1)
           # bids_o_matic_9000(raw_dicom[0],gear_context)

            gear_context.log.debug('Successfully generated BIDS compliant files')

if __name__ == '__main__':
    main()
