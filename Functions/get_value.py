import numpy as np

def get_value(value):
    if isinstance(value, np.floating) == False:
        if len(value)>1:
            return value[0]
        elif len(value)==0: 
            return None
    else:
        return value
        