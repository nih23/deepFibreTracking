'Utility function and classes'
from string import Formatter
import torch

def info(text):
    """Print simple Info text.
    
    Arguments:
    text: the text to print
    """
    print("[INFO] {}".format(text) + ' ' * 150)
def progress(percentage, text="", prefix="INFO"):
    """Simple Progressbar to show progress in terminal.
    
    Arguments:
    percentage: a progress percentage, between 0 and 100
    text: optional info text describing things being done
    prefix: prefix, default:  'INFO'
    """
    percent = ("{0:.1f}").format(percentage)
    filledLength = int(percentage)
    bar = '=' * filledLength + '-' * (100 - filledLength)

    print("\r[%s] %s |%s| %s%% " % (prefix,text,bar, percent), end='\r')
def radians_loss(input_data, target, mask):
    """Quick implementation of the radian loss 1- cos(alpha). 
    
    Arguments:
    input_data: the network output
    target: the supposed output
    mask: mask for masking out unused padding areas. Essential to prevent division by zero."""
    cossim = torch.nn.CosineSimilarity(dim=2)
    output = cossim(input_data, target)**2
    output = output[mask.squeeze() != 0]
    return 1 - torch.mean(output)

def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Taken from: https://stackoverflow.com/questions/8906926/formatting-timedelta-objects/17847006#17847006
    
    Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    formatter = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in formatter.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return formatter.format(fmt, **values)
