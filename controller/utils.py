import datetime
from .skill_item import SKILL_RET_TYPE

def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)

def input_t(literal):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    return input(f"[{current_time}] {literal}")

def evaluate_value(s: str) -> SKILL_RET_TYPE:
    if s.isdigit():
        return int(s)
    elif s.replace('.', '', 1).isdigit():
        return float(s)
    elif s == 'True':
        return True
    elif s == 'False':
        return False
    elif s == 'None' or len(s) == 0:
        return None
    else:
        if not (s.startswith("'") and s.endswith("'")):
            return f"'{s}'"
        return s