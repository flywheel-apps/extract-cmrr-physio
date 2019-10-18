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

##-------- Standard Flywheel Gear Structure --------##
flywheelv0 = "/flywheel/v0"
environ_json = '/tmp/gear_environ.json'

##--------  Gear Specific files/folders/values  ----##
tic_len = 2.5e-3  # seconds, length of one "tick"


def physio_qc(physio_dict,new_vol_tics,output_dir):

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

    # Now ensure that the physio values are a plottable format
    physio_vals = np.array(physio_dict['VALUE']).astype(int)

    # Extract the name of the physiological measure
    phys_name = physio_dict['LogDataType']

    # Generate a subplot and set up some aesthetics
    f, ax = pl.subplots(1, 1, figsize=(10, 4))
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot the physio data
    ax.plot(physio_time, physio_vals, linewidth=0.7, label='physio {}'.format(phys_name))

    # Get the natural y axis limits and plot two vertical lines, indicating the start and stop of the scan
    yl, yh = ax.get_ylim()
    ax.axvline(vol_time[0],yl,yh,color='g',linewidth=1.5,label='Volume Acquisition start')

    # Since our new_vol measurements are the time of the start of a volume acquisition, the true end of the scan
    # is one TR after the last acquisition start.  So, we can just calculate a "dt" and modify our end time
    dt = np.mean(np.diff(vol_time))
    ax.axvline(vol_time[-1]+dt,yl,yh,color='r',linewidth=1.5,label='Volume Acquisition end')

    # Set the plot layout to tight and add a legend
    pl.tight_layout()
    pl.legend()

    # Save the figure and close
    pl.savefig(op.join(output_dir, '{}_physio.png'.format(phys_name)), dpi=250)
    pl.close()

    pass


def run_qc_on_physio_logs(context):
    """
    This function runs the function to generate quality control measures (graphs) of the physio output
    :param context:
    :return:
    """
    # First extract the info, we need new_vol_tics
    info = context.custom_dict['physio-dicts']['info']
    new_vol_tics = info['NEW_VOL_TICS']

    for physio in context.custom_dict['physio-dicts']:

        # If this dict is info, skip
        if physio == 'info':
            continue

        # Otherwise extract the dict and pass that to the qc function
        physio_dict = context.custom_dict['physio-dicts']['physio']
        physio_qc(physio_dict, new_vol_tics)

    return

def log2dict(physio_log,context=''):
    physio_dict = {}
    header = []
    #context.log.debug('loading file')
    f = open(physio_log, 'r')
    #context.log.debug('going through lines')
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

    return physio_dict



def create_physio_dicts_from_logs(context):

    """
    This function generates easily accessible dictionary objects for each physio log file

    :param context: The gear context
    :return:
    """

    # Set the output dir
    physio_output_dir = context.get_output_dir()

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

    for physio in all_physio:

        # If this in the info file, we don't do it.
        if physio == info:
            continue

        # Otherwise make it a dictionary:
        physio_dict = log2dict(physio, context)
        context.log.debug('complete')
        physio_dict['ACQ_TIME_TICS'] = np.array(physio_dict['ACQ_TIME_TICS']).astype(int)
        physio_dict['VALUE'] = np.array(physio_dict['VALUE']).astype(int)

        # Store this as a dictionary in our context custom dict
        context.custom_dict['physio-dicts'][physio_dict['LogDataType']] = physio_dict

    # The dictionaries are now stored in context.  We can return to the calling function
    return



def dicts2bids(physio_dict, new_vol_ticks):

    # Create an empty bids file.  this has the physio value and the scan trigger
    phys_tics = physio_dict['ACQ_TIME_TICS']
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


def bids_o_matic_9000(physio_output_dir, raw_dicom, context):
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

    # See if the "info" tag is filled and has the "Series Description" key:
    matches = context.get_input('DICOM_ARCHIVE')['object']['info']
    if 'SeriesDescription' in matches.keys() and matches['SeriesDescription']:

        # Use that
        matches = matches['SeriesDescription']

    else:
        # If not, we'll load the dicom and find something from there...
        # But first warn the user
        context.log.warning('dicom metadata missing values.  Extracting "protocol name" from dicom for bids naming')
        context.log.debug('loading {}'.format(raw_dicom))
        dicom = pydicom.read_file(raw_dicom)
        context.log.debug('dicom protocol name: {}'.format(dicom.ProtocolName))

        # If the protocol name is empty in the dicom, then this is all just messed up
        if not dicom.ProtocolName:
            context.log.warning('Dicom header is missing values.  Unable to properly name BIDs physio file')
            context.log.warning('User will need to manually rename')

            # Let the user manually assign the name later
            matches = 'UnknownAcquisition'
        else:
            # Otherwise use what's provided in the dicom's Protocol name
            context.log.debug('Settin matchis in the dicom protocol else loop')
            matches = '{}'.format(dicom.ProtocolName)

    context.log.info('matches value:')
    context.log.info('matches: {}'.format(matches))
    context.log.debug('dicom protocol name: {}'.format(dicom.ProtocolName))
    context.log.debug('matches type: {}'.format(type(matches)))



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
        if label == "RESP":
            label = 'respiration'
        elif label == "PULS":
            label = 'cardiac'

        # Find the sample interval (in tics)
        json_dict['SamplingFrequency'] = 1.0 / float(physio_dict['SampleTime']) * tic_len

        # Set the file name and save the file
        physio_output = op.join(context.output_dir,'{}_recording-{}_physio.tsv'.format(matches,label))
        np.savetxt(physio_output, bids_file, fmt='%d', delimiter='\t')

        # Zip the file
        gzip_command = ['gzip',physio_output]
        exec_command(context, gzip_command)

        # Now create and save the .json file
        json_output = op.join(context.output_dir,'{}_recording-{}_physio.json'.format(matches,label))
        json_dict['StartTime'] = (physio_dict['ACQ_TIME_TICS'][0] - info_dict['ACQ_START_TICS'][0]) * tic_len
        json_dict['Columns'] = [label,'trigger']

        with open(json_output,'w') as json_file:
            json.dump(json_dict, json_file)


        # EASY PEASY LIVIN GREEZY

    pass




def main():
    shutil.copy('config.json','/flywheel/v0/output/config.json')
    with flywheel.gear_context.GearContext() as gear_context:

        #### Setup logging as per SSE best practices (Thanks Andy!)
        fmt = '%(asctime)s %(levelname)8s %(name)-8s - %(message)s'
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

        fu.exists(dicom, gear_context.log, '.dicom.zip')

        # Now we need to unzip it:
        uz_dir = op.join(flywheelv0, 'unzipped_dicom')
        unzip_dicom_command = ['unzip', dicom, '-d', uz_dir]

        ###########################################################################
        # Try to run the unzip command:

        try:
            exec_command(gear_context, unzip_dicom_command)
        except Exception as e:
            gear_context.log.fatal(e, )
            gear_context.log.fatal('The CMRR physio extraction Failed.', )
            os.sys.exit(1)

        # Now we locate the raw unzipped dicom
        raw_dicom = glob.glob(op.join(uz_dir, '*.dicom', '*.dcm'))

        if len(raw_dicom) > 1:
            gear_context.log.fatal(
                'Dicom structure contains too many dicoms, unrecognized CMRR physio format for this gear')

            sys.exit(1)
        elif len(raw_dicom) < 1:
            gear_context.log.fatal('Dicom structure unzipped zero files.')
            sys.exit(1)

        run_physio_command = ['/usr/local/bin/extractCMRRPhysio', raw_dicom[0], output_dir]

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
        # Try to generate physio dictionaries from logs (used for QC and Bids)
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
            create_physio_dicts_from_logs(gear_context)
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
                gear_context.log.fatal('The CMRR conversion to bids failed', )

                os.sys.exit(1)

            gear_context.log.debug('Successfully generated BIDs compliant files')

if __name__ == '__main__':
    main()
