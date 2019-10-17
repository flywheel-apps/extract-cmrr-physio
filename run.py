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

##-------- Standard Flywheel Gear Structure --------##
flywheelv0 = "/flywheel/v0"
environ_json = '/tmp/gear_environ.json'

##--------    Gear Specific files/folders   --------##
tic_len = 2.5e-3  # seconds, length of one "tick"


def log2dict(physio_log):
    physio_dict = {}
    header = []
    f = open(physio_log, 'r')

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

        # Otherwise, if we remove the spaces and it's NOT numeric, then it's headers and we need it
        elif not line.replace(' ', '').isnumeric():
            header = line.split()
            for h in header:
                physio_dict[h] = []

        # Otherwise, if we remove the space and it IS numeric, it's numbers and we need it
        elif line.replace(' ', '').isnumeric():

            # if headers are empty, we hit numbers before we hit headers
            if not header:
                raise Exception('Unable to parse .log files, no headers found')

            vals = line.split()
            for h, v in zip(header, vals):
                physio_dict[h].append(v)

        else:
            print('Unrecognized line: {}\nskipping'.format(line))

    return physio_dict


def dicts2bids(physio_dict, new_vol_ticks):
    # Create an empty bids file.  this has the physio value and the scan trigger
    phys_tics = physio_dict['ACQ_TIME_TICS']
    phys_valu = physio_dict['VALUE']
    bids_file = np.zeros((len(phys_valu, 2)))

    vol = 0

    for i, tic in enumerate(phys_tics):

        bids_file[i, 0] = phys_valu[i]

        # If we're not at the last volume
        if vol > len(new_vol_ticks):

            # Check if the current tic is greater than or equal to the next volume tic,
            # Then this is as close as we can get to a scanner trigger, so set it as 1
            next_vol_tick = new_vol_ticks[vol + 1]
            if tic >= next_vol_tick:
                bids_file[i, 1] = 1

                # and increment the volume
                vol += 1

    return bids_file


def bids_o_matic_9000(physio_output_dir, context):
    # First let's check the output.  We'll try to make this robust for any measurement:
    all_physio = glob.glob(op.join(physio_output_dir, '*.log'))
    info = glob.glob(op.join(physio_output_dir, '*Info.log'))

    # Grab useful information from the input object:
    # variable names are following bids tutorial:
    # https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continous-recordings.html
    matches = context.get_input('DICOM_ARCHIVE')['object']['info']['SeriesDescription']

    # Handle errors and warnings about missing physio data
    if len(all_physio) == 0 or (len(all_physio) == 1 and len(info) == 1):
        context.log.warning('No physio signals extracted')
        return

    # First we need to load the acquisition info file, and look through to see when the volumes were acquired
    # I believe we need this for the "trigger".  Typically, the scanner triggers at every volume.  I'm ASSUMING
    # That's what they're looking for, but it's not completely clear.
    info_dict = log2dict(info)
    json_dict = {}

    # Let's convert the "volume" and "ticks" array to int
    info_dict['VOLUME'] = np.array(info_dict['VOLUME']).astype(int)
    info_dict['ACQ_START_TICS'] = np.array(info_dict['ACQ_START_TICS']).astype(int)

    # Take the difference to see when it changes and extract the ticks at those changes
    new_vol_inds = np.where(np.diff(info_dict['VOLUME']))[0] + 1
    new_vol_ticks = info_dict['ACQ_START_TICS'][new_vol_inds]

    for physio in all_physio:

        # If this in the info file, we don't do it.
        if physio == info:
            continue

        # Otherwise make it a dictionary:
        physio_dict = log2dict(physio)
        physio_dict['ACQ_TIME_TICS'] = np.array(physio_dict['ACQ_START_TICS']).astype(int)
        physio_dict['VALUE'] = np.array(physio_dict['VALUE']).astype(int)

        # Make a bids compatible array
        bids_file = dicts2bids(physio_dict, new_vol_ticks)

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
    # shutil.copy('config.json','/flywheel/v0/output/config.json')
    with flywheel.gear_context.GearContext() as gear_context:

        #### Setup logging as per SSE best practices (Thanks Andy!)
        fmt = '%(asctime)s %(levelname)8s %(name)-8s - %(message)s'
        logging.basicConfig(level=gear_context.config['gear-log-level'], format=fmt)

        gear_context.log = logging.getLogger('[flywheel/extract-cmrr-physio]')

        gear_context.log.info('log level is ' + gear_context.config['gear-log-level'])

        gear_context.log_config()  # not configuring the log but logging the config

        # Now let's set up our environment from the .json file stored in the docker image:
        environ = fu.set_environment(gear_context.log)
        output_dir = gear_context.output_dir

        # Now we need to extract our input file, and check if it exists
        dicom = gear_context.get_input_path('DICOM_ARCHIVE')
        run_bids = gear_context.config['Generate Bids']

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

        if run_bids:

            ###########################################################################
            # Try to run the bidsify command:
            try:
                bids_o_matic_9000(output_dir,gear_context)
            except Exception as e:
                gear_context.log.fatal(e, )
                gear_context.log.fatal('The CMRR conversion to bids failed', )

                os.sys.exit(1)