import os
import sys
import json
import logging



def exists(file, ext=-1, is_expected=True, quit_on_error=True):

    # Generic if_exists function that takes care of logging in the event of nonexistance
    # is_expected indicates if we're checking to see if it's there or not. True: we want it to be there, false: we don't
    # quit_on_error tells us if we sys.exit on failure or not.
    log = logging.getLogger()

    path_exists=os.path.exists(file)

    # If we find the file and are expecting to
    if path_exists and is_expected:
        log.info('located {}'.format(file))

    # If we don't find the file and are expecting to
    elif not path_exists and is_expected:
        # and if that file is critical
        if quit_on_error:
            # Quit the program
            log.error('Unable to locate {} '.format(file))
            sys.exit(1)
            # Otherwise, we'll manage.  Keep on trucking.
        else:
            log.warning('Unable to locate {} '.format(file))

    # If we don't find the file and we weren't expecting to:
    elif not path_exists and not is_expected:
        # Then we're all good, keep on trucking
        log.info('{} is not present or has been removed successfully'.format(file))

    # If we don't expect the file to be there, but it is...DUN DUN DUNNNN
    elif path_exists and not is_expected:
        # and if that file is critical
        if quit_on_error:
            # Well, you know the drill by now.
            log.error('file {} is present when it must be removed'.format(file))
            sys.exit(1)
        else:
            log.warning('file {} is present when it should be removed'.format(file))

    # Now we'll check the file extension (if desired)
    if isinstance(ext, str):

        num_periods = ext.count('.')
        file_name = os.path.split(file)[-1]
        split_by_period = file_name.split('.')

        # I know, it's ugly.  but it works.
        file_ext = '.'+'.'.join(split_by_period[-num_periods:])

        if not file_ext == ext:
            log.error('Incorrect file type for input {}, expected {}, got {}'.format(file, ext, file_ext))
            sys.exit(1)

    return path_exists


def set_environment(environ_json, log):

    # Let's ensure that we have our environment .json file and load it up
    exists(environ_json, log)

    # If it exists, read the file in as a python dict with json.load
    with open(environ_json, 'r') as f:
        log.info('Loading gear environment')
        environ = json.load(f)

    # Now set the current environment using the keys.  This will automatically be used with any sp.run() calls,
    # without the need to pass in env=...  Passing env= will unset all these variables, so don't use it if you do it
    # this way.
    for key in environ.keys():
        os.environ[key] = environ[key]

    # Pass back the environ dict in case the run.py program has need of it later on.
    return environ
