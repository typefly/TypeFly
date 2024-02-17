from typing import Union
import datetime

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

def split_args(arg_str):
        args = []
        current_arg = ''
        parentheses_count = 0  # Keep track of open parentheses

        for char in arg_str:
            if char == ',' and parentheses_count == 0:
                # If we encounter a comma and we're not inside parentheses, split here
                args.append(current_arg.strip())
                current_arg = ''
            else:
                # Otherwise, keep adding characters to the current argument
                if char == '(':
                    parentheses_count += 1
                elif char == ')':
                    parentheses_count -= 1
                current_arg += char

        # Don't forget to add the last argument after the loop finishes
        if current_arg:
            args.append(current_arg.strip())

        return args