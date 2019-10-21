import os, os.path as op
import subprocess as sp
import re

def BuildCommandList(command, ParamList):
    """
    command is a list of prepared commands
    ParamList is a dictionary of key:value pairs to be put into the command list as such ("-k value" or "--key=value")
    """
    for key in ParamList.keys():
        # Single character command-line parameters are preceded by a single '-'
        if len(key) == 1:
            command.append('-' + key)
            if len(str(ParamList[key]))!=0:
                command.append(str(ParamList[key]))
        # Multi-Character command-line parameters are preceded by a double '--'
        else:
            # If Param is boolean and true include, else exclude
            if type(ParamList[key]) == bool:
                if ParamList[key]:
                    command.append('--' + key)
            else:
                # If Param not boolean, but without value include without value
                # (e.g. '--key'), else include value (e.g. '--key=value')
                if len(str(ParamList[key])) == 0:
                    command.append('--' + key)
                else:
                    command.append('--' + key + '=' + str(ParamList[key]))
    return command


def exec_command(context, command, shell=False, stdout_msg=None):
    context.log.info('Executing command: \n' + ' '.join(command)+'\n\n')
    if not context.custom_dict['dry-run']:
        # The 'shell' parameter is needed for bash output redirects 
        # (e.g. >,>>,&>)
        if shell:
            command = ' '.join(command)
        result = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE,
                        universal_newlines=True, shell=shell)

        stdout, stderr = result.communicate()
        context.log.info('Command return code: {}'.format(result.returncode))

        if stdout_msg==None:
            context.log.info(stdout)
        else:
            context.log.info(stdout_msg)

        if result.returncode != 0:
            context.log.error('The command:\n ' +
                              ' '.join(command) +
                              '\nfailed.')
            raise Exception(stderr)

def set_subject(context):
    """
    This function queries the subject from the session only if the 
    context.config['Subject'] is invalid or not present.
    Exits ensuring the value of the subject is valid
    """

    if 'Subject' in context.config.keys():
        # Correct for non-friendly characters
        subject = re.sub('[^0-9a-zA-Z./]+', '_', context.config['Subject'])
        if len(subject) > 0:
            context.config['Subject'] = subject
            return 

    # Assuming valid client
    fw = context.client
    # Get the analysis destination ID
    dest_id = context.destination['id']
    # Assume that the destination object has "subject" as a parent
    dest = fw.get(dest_id)
    subj = fw.get(dest.parents['subject'])
    subject = subj.label
    # return the label of that subject
    context.config['Subject'] = subject
    context.log.info('Using {} as Subject ID.'.format(subject))

