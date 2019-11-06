#!/usr/bin/env python3

import flywheel
from utils import futils as fu
from utils import physio as ph
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
from pprint import pprint as pp
import copy
from collections import Counter
import utils.physio as phys

##-------- Standard Flywheel Gear Structure --------##
flywheelv0 = "/flywheel/v0"
environ_json = '/tmp/gear_environ.json'

##--------  Gear Specific files/folders/values  ----##
tic_len = 2.5e-3  # seconds, length of one "tick"

def data_classifier(context,output_dir):

    """
    This function takes the gear context and the output directory and loads the "physio_dicts" from the context
    custom dicts object.  Each physio dict has a key "['bids_file']" and "['log_file'].  if these files were created,
    these keys point to that filename.  Loop through the log files and the bids files and set metadata based off
    dictionary parameters and inhereted properties.
    :param context: gear context
    :param output_dir: the directory to save the metadata file in.
    :return:
    """

    custom_physio_class = "Physio"
    custom_ext_class = "Trigger"
    custom_ecg_class = "ECG"
    custom_info_class = "Info"
    inputs = context._get_invocation()['inputs']
    image_info = inputs['DICOM_ARCHIVE']['object']['info']

    # Check if physio is in the input object's information anywhere:
    imtype = image_info['ImageType']
    if not any([i=='PHYSIO' for i in imtype]):
        context.log.warning('ImageType does not indicate physio, however by virtue of the gear running successfully, we will assume physio type')



    # Attempt to recover classification info from the input file
    (config, modality, classification) = ([], None, [])
    try:
        classification = inputs['DICOM_ARCHIVE']['object']['classification']
        classification['Custom'] =['Physio']
    except:
        context.log.info('  Cannot determine classification from config.json.')
        classification = {'Custom':['Physio']}
    try:
        modality = inputs['DICOM_ARCHIVE']['object']['modality']
    except:
        context.log.info('  Cannot determine modality from config.json.')
        modality = 'MR'

    files = []
    for physio in context.custom_dict['physio-dicts']:

        # Extract the dictionary
        physio_dict = context.custom_dict['physio-dicts'][physio]
        label = physio_dict['LogDataType']

        # Now we'll determine the custom classification
        if label == 'EXT':
            classification['Custom'] = [custom_ext_class, custom_physio_class]
            modality = 'MR'
            context.log.debug('setting custom info for trigger')

        elif label == 'ECG':
            modality = 'ECG'
            classification['Custom'] = [custom_ecg_class, custom_physio_class]
            context.log.debug('setting custom info for ECG')

        elif label == 'ACQUISITION_INFO':
            classification['Custom'] = [custom_info_class, custom_physio_class]
            context.log.debug('setting custom info for acquisition info')

        else:
            classification['Custom'] = [custom_physio_class]
            modality = 'MR'
            context.log.debug('setting custom info for physio')


        # This will loop through each physio_dict's log and bids keys
        for file_key in ['log_file', 'bids_tsv', 'bids_json']:

            # Label the log files
            if physio_dict[file_key]:

                # First set the filetype:
                f = physio_dict[file_key]

                if f.endswith('.qa.png'):
                    ftype = 'qa'

                elif f.endswith('.log'):
                    ftype = 'log'

                elif f.endswith('.tsv.gz'):
                    ftype = 'tabular data'

                elif f.endswith('.json'):
                    ftype = 'json'

                else:
                    ftype = 'unknown'




            fdict = {'name': f,
                     'type': ftype,
                     'classification': copy.deepcopy(classification),
                     'info': image_info,
                     'modality': modality}

            files.append(fdict)

            # Print info to log
            context.log.info('file:\t{}\n'.format(f) +
                             'type:\t{}\n'.format(ftype) +
                             'classification:\t{}\n'.format(pp(classification)) +
                             'modality:\t{}\n'.format(modality))

    # Collate the metadata and write to file
    metadata = {}
    metadata['acquisition'] = {}
    metadata['acquisition']['files'] = files
    #pp(metadata)

    metadata_file = os.path.join(output_dir, '.metadata.json')
    with open(metadata_file, 'w') as metafile:
        json.dump(metadata, metafile)

    metadata_file = os.path.join(output_dir, 'metaout.json')
    with open(metadata_file, 'w') as metafile:
        json.dump(metadata, metafile)





def main():

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

        # Set up a field for physio dictionaries (used for bidsifying and qc)
        gear_context.custom_dict['physio-dicts'] = {}

        # Now let's set up our environment from the .json file stored in the docker image:
        fu.set_environment(environ_json, gear_context.log)
        output_dir = gear_context.output_dir

        # Now we need to extract our input file, and check if it exists
        dicom = gear_context.get_input_path('DICOM_ARCHIVE')
        run_bids = gear_context.config['Generate_Bids']

        matches = gear_context.get_input('DICOM_ARCHIVE')['object']['info']
        if 'SeriesDescription' in matches.keys() and matches['SeriesDescription']:
            # Use that
            matches = matches['SeriesDescription']
        else:
            matches = ''

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
        # Get the output.log files and make the physio objects
        # Set the output dir
        physio_output_dir = gear_context.output_dir

        # First let's check the output.  We'll try to make this robust for any measurement:
        gear_context.log.debug('Globbing Physio')
        all_physio = glob.glob(op.join(physio_output_dir, '*.log'))
        info_file = glob.glob(op.join(physio_output_dir, '*Info.log'))[0]

        # Handle errors and warnings about missing physio data
        if len(all_physio) == 0 or (len(all_physio) == 1 and len(info) == 1):
            gear_context.log.warning('No physio signals extracted')
            return

        # First we need to load the acquisition info file, and look through to see when the volumes were acquired
        # I believe we need this for the "trigger".  Typically, the scanner triggers at every volume.  I'm ASSUMING
        # That's what they're looking for, but it's not completely clear.
        info = phys.phys_info(info_file)
        physio_objects = []

        for phys_file in all_physio:

            # If it's the info file, we just made that object, so skip it.
            if phys_file == info_file:
                continue
            physio_objects.append(phys.physio(phys_file))

        for physio in physio_objects:
            try:
                # Set the info object
                physio.set_info(info)
                physio.parent_dicom = gear_context.get_input_path('DICOM_ARCHIVE')

                # Set some values that in the future will be set by the config file
                physio.fill_val = 0
                physio.interp = 'linear'
                physio.handle_missing_data = 'uniform'

                # Mandatory Processing step:
                physio.remove_duplicate_tics()

                # Based on the handle_missing_data, process the data
                process = gear_context.config['process data']
                if gear_context.config['process data']:
                    physio.create_new_tic_array()
                    physio.interp_values_to_newtics()
                    physio.triggers_2_timeseries()


                #Run QA:
                physio.run_full_qc()

                # If BIDS is desired:
                if run_bids:
                    physio.bids_o_matic_9000(processed=process, matches=matches, zip_output=False)

            except Exception as e:
                gear_context.log.fatal(e)
                sys.exit(1)



        ###########################################################################
        # If the user doesn't want to keep these log files, delete them.
        if not gear_context.config['Generate_Raw']:
            gear_context.log.info('Removing .log files')
            cmd = ['/bin/rm', output_dir,'*.log']
            try:
                exec_command(cmd)
            except Exception as e:
                gear_context.log.warning(e, )
                gear_context.log.warning('Unable to remove *.log files', )


if __name__ == '__main__':
    main()


