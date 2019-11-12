#!/usr/bin/env python3
import flywheel
from utils import futils as fu
import logging
from utils.Common import exec_command
import os.path as op
import os
import glob
import sys
import json
from pprint import pprint as pp
import copy
import utils.physio as phys

##-------- Standard Flywheel Gear Structure --------##
flywheelv0 = "/flywheel/v0"
environ_json = '/tmp/gear_environ.json'


def data_classifier(context,physio_dict,file):


    """
    This function takes the gear context and the output directory and loads the "physio_dicts" from the context
    custom dicts object.  Each physio dict has a key "['bids_file']" and "['log_file'].  if these files were created,
    these keys point to that filename.  Loop through the log files and the bids files and set metadata based off
    dictionary parameters and inhereted properties.
    :param context: gear context
    :type context: Context
    :param output_dir: the directory to save the metadata file in.
    :type output_dir: str
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
    (config, modality, classification) = ([], None, {})
    try:
        classification = inputs['DICOM_ARCHIVE']['object']['classification']
        classification['Custom'] = ['Physio']
    except:
        context.log.info('  Cannot determine classification from config.json.')
        classification = {'Custom':['Physio']}
    try:
        modality = inputs['DICOM_ARCHIVE']['object']['modality']
    except:
        context.log.info('  Cannot determine modality from config.json.')
        modality = 'MR'


    # Extract the dictionary
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



    if file.endswith('.qa.png'):
        ftype = 'qa'

    elif file.endswith('.log'):
        ftype = 'log'

    elif file.endswith('.tsv.gz') or file.endswith('.tsv'):
        ftype = 'tabular data'

    elif file.endswith('.json'):
        ftype = 'json'

    else:
        ftype = 'unknown'




    fdict = {'name': file,
             'type': ftype,
             'classification': copy.deepcopy(classification),
             'info': image_info,
             'modality': modality}


    # Print info to log
    context.log.info('file:\t{}\n'.format(file) +
                     'type:\t{}\n'.format(ftype) +
                     'classification:\t{}\n'.format(pp(classification)) +
                     'modality:\t{}\n'.format(modality))


    context.update_file_metadata(file, fdict)



def setup_logger(gear_context):
    """
    This function simply sets up the gear logger to Flywheel SSE best practices
    :param gear_context: the gear context
    :return: none
    """
    #### Setup logging as per SSE best practices
    fmt = '%(asctime)s %(levelname)8s %(name)-8s %(funcName)s - %(message)s'
    logging.basicConfig(level=gear_context.config['gear-log-level'], format=fmt)
    gear_context.log = logging.getLogger('[flywheel/extract-cmrr-physio]')
    gear_context.log.info('log level is ' + gear_context.config['gear-log-level'])
    gear_context.log_config()  # not configuring the log but logging the config


def extract_zipped_dicom(gear_context, dicom):
    """
    This function extracts a zipped dicom archive and check for the correct number of resulting files for a physio
    DICOM archive.  Errors are thrown if there's more or less than 1 dicom file extracted, or if the unzipped file
    is not of the supported type.
    :param gear_context: the gear context
    :param dicom: the zipped dicom file archive
    :return: the raw, unzipped file path
    """

    try:
        supported_filetypes = [".dcm", ".IMA"]

        fu.exists(dicom, gear_context.log, '.zip')
        zip_base = op.splitext(op.split(dicom)[-1])[0]

        # Now we need to unzip it:
        uz_dir = op.join('/tmp', 'unzipped_dicom')
        unzip_dicom_command = ['unzip', '-o', dicom, '-d', uz_dir]
        ###########################################################################
        # Try to run the unzip command:
        try:
            exec_command(gear_context, unzip_dicom_command)
        except Exception as e:
            raise e

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
            raise Exception('Too many dicoms in Zip file archive')

        # IF we didn't find any, exit
        elif len(raw_dicom) < 1:
            gear_context.log.fatal('Dicom structure unzipped zero file of supported file type.')
            raise Exception('No files found in unzipped archive')

    except Exception as e:
        raise e

    return raw_dicom

def physio_json_2_bids_metadata(context,bids_file,json):
    """
    This function updates the info section of the metadata for file bids_file with info found in the json dict.
    :param context: gear context
    :type context: class: `flywheel.gear_context.GearContext`
    :param bids_file: filename to update the metadata of
    :param json: the json dictionary with the necessary keys
    :return:
    """

    update_metadata = {
        "info": {
            "SamplingFrequency": json['SamplingFrequency'],
            "StartTime": json['StartTime'],
            "Columns": json['Columns']
        }
    }

    context.update_file_metadata(bids_file, update_metadata)



def main():

    with flywheel.gear_context.GearContext() as gear_context:

        try:

            ######################################################
            ####   Runtime Setup
            ######################################################
            # Setup gear logging
            setup_logger(gear_context)

            # Extract some basic values and set up gear environment
            output_dir = gear_context.output_dir
            fu.set_environment(environ_json, gear_context.log)

            # Set up Custom Dicionary to host user variables
            gear_context.custom_dict = {}
            gear_context.custom_dict['environ'] = environ_json

            # Create a 'dry run' flag for debugging
            gear_context.custom_dict['dry-run'] = gear_context.config['Dry-Run']

            # Set up a field for physio dictionaries (used for bidsifying and qc)
            gear_context.custom_dict['physio-dicts'] = {}

            # Extract matches keyword for BIDS data format
            matches = gear_context.get_input('DICOM_ARCHIVE')['object']['info']
            if 'SeriesDescription' in matches.keys() and matches['SeriesDescription']:
                # Use that
                matches = matches['SeriesDescription']
            else:
                matches = ''

            ######################################################
            ####   Pre-processing/file setup/extraction
            ######################################################
            # Now we need to unzip our input file, and check if it exists
            dicom = gear_context.get_input_path('DICOM_ARCHIVE')
            raw_dicom = extract_zipped_dicom(gear_context, dicom)

            #########################################
            # Try to run the extract physio command:
            run_physio_command = ['/usr/local/bin/extractCMRRPhysio', raw_dicom[0], output_dir]
            gear_context.custom_dict['raw_dicom'] = raw_dicom[0]
            exec_command(gear_context, run_physio_command)

            gear_context.log.debug('Successfully extracted physio')

            #########################################
            # Get the output.log files and make the physio objects
            # Set the output dir
            physio_output_dir = gear_context.output_dir

            # First let's check the output.  We'll try to make this robust for any measurement:
            gear_context.log.debug('Globbing Physio')
            all_physio = glob.glob(op.join(physio_output_dir, '*.log'))
            info_file = glob.glob(op.join(physio_output_dir, '*Info.log'))

            if not info_file:
                gear_context.log.warning('No Info file found.  Failed Extraction?')
                return
            else:
                info_file = info_file[0]

            # Handle errors and warnings about missing physio data
            if len(all_physio) == 0 or len(all_physio) == 1:
                gear_context.log.warning('No physio signals extracted')
                return

            ######################################################
            ####   Main Physio Processing/BIDS creation
            ######################################################
            info = phys.phys_info(info_file)
            physio_objects = []

            for physio_file in all_physio:

                try:
                    data_classifier(gear_context, physio.physio_dict, physio.logfile)
                except Exception as e:
                    gear_context.log.info("error updating metadata in log")
                    gear_context.log.exception(e)

                # If it's the info file, we just made that object, so skip it.
                if physio_file == info_file:
                    continue

                physio = phys.physio(physio_file)

                # Set the info object
                physio.set_info(info)
                physio.parent_dicom = raw_dicom[0]

                # Set some values that will be set by the config file
                physio.fill_val = gear_context.config['Fill_Value']
                physio.interp_method = gear_context.config['Interpolation_Method']
                physio.tic_fill_strategy = gear_context.config["Missing_Data"]

                # Mandatory Processing step:
                physio.remove_duplicate_tics()

                # Based on the handle_missing_data, process the data
                process = gear_context.config['Process_Data']
                if process:
                    physio.create_new_tic_array()
                    physio.interp_values_to_newtics()
                    physio.triggers_2_timeseries()

                #Run QA:
                physio.run_full_qc()

                # If BIDS is desired:
                if gear_context.config['Generate_Bids']:
                    physio.bids_o_matic_9000(processed=process, matches=matches, zip_output=False, save_json=gear_context.config['Generate_json'])

                    physio_json_2_bids_metadata(gear_context, physio.bids_file, physio.bids_json)
                    try:
                        data_classifier(gear_context, physio.physio_dict, physio.bids_file)
                    except Exception as e:
                        gear_context.log.info("error updating metadata in BIDS file")
                        gear_context.log.exception(e)
                #physio_objects.append(phys.physio(physio))


            ###########################################################################
            # If the user doesn't want to keep these log files, delete them.
            if not gear_context.config['Generate_Raw']:
                gear_context.log.info('Removing .log files')
                cmd = ['/bin/rm', output_dir,'*.log']

                exec_command(cmd)

        # Catch any exceptions
        except Exception as e:
            gear_context.log.exception(e)
            sys.exit(1)

if __name__ == '__main__':
    main()

## TODO:
# Add data quality flag for large gaps and lots of high/low cutoff values

