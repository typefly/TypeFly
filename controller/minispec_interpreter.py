from typing import List, Tuple, Union
import re

from .skillset import SkillSet
from .utils import split_args

MiniSpecValueType = Union[int, float, bool, str, None]

def evaluate_value(value) -> MiniSpecValueType:
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit():
        return float(value)
    elif value == 'True':
        return True
    elif value == 'False':
        return False
    elif value == 'None' or len(value) == 0:
        return None
    else:
        return value.strip('\'"')

class MiniSpecReturnValue:
    def __init__(self, value: MiniSpecValueType, replan):
        self.value = value
        self.replan = replan

    def from_tuple(t: Tuple[MiniSpecValueType, bool]):
        return MiniSpecReturnValue(t[0], t[1])
    
    def default():
        return MiniSpecReturnValue(None, False)
    
    def __repr__(self) -> str:
        return f'value={self.value}, replan={self.replan}'

class MiniSpecInterpreter:
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self):
        self.env = {}
        if MiniSpecInterpreter.low_level_skillset is None or \
            MiniSpecInterpreter.high_level_skillset is None:
            raise Exception('MiniSpecInterpreter: Skillset is not initialized')

    def get_env_value(self, var) -> MiniSpecValueType:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]
    
    # def check_statement(self, code):
    #     message = []
    #     if code.count('{') != code.count('}'):
    #         message.append('Syntax error: unbalanced brackets')

    #     statements = self.split_statements(code)
    #     for statement in statements:
    #         message = message + self.check_statement(statement)

    def split_statements(self, code) -> List[str]:
        statements = []
        stack = []
        start = 0
        for i, char in enumerate(code):
            if char == '{':
                stack.append('{')
            elif char == '}':
                stack.pop()
                if len(stack) == 0:
                    statements.append(code[start:i+1].strip())
                    start = i + 1
            elif char == ';' and len(stack) == 0:
                statements.append(code[start:i].strip())
                start = i + 1

        if start < len(code):
            statements.append(code[start:].strip())

        return [s for s in statements if s]

    def execute(self, code) -> MiniSpecReturnValue:
        statements = self.split_statements(code)
        for statement in statements:
            print(f'Executing statement: {statement}')
            val = None
            replan = False
            if not statement:
                continue
            elif statement.startswith('->'):
                # return statement, won't trigger replan
                return self.evaluate_return(statement)
            elif re.match(r'^\d', statement):
                # loop statement, may trigger replan
                ret_val = self.execute_loop(statement)
            elif statement.startswith('?'):
                # conditional statement, may trigger replan
                ret_val = self.execute_conditional(statement)
            else:
                # function call, may trigger replan
                ret_val = self.execute_function_call(statement)

            if ret_val.replan:
                return ret_val
        return MiniSpecReturnValue.default()

    def evaluate_return(self, statement) -> MiniSpecReturnValue:
        _, value = statement.split('->')
        if value.startswith('_'):
            value = self.get_env_value(value)
        else:
            value = evaluate_value(value.strip())
        return MiniSpecReturnValue(value, False)
    
    def execute_loop(self, statement) -> MiniSpecReturnValue:
        count, program = re.match(r'(\d+)\s*\{(.+)\}', statement).groups()
        for i in range(int(count)):
            print(f'Executing loop iteration {i}')
            ret_val = self.execute(program)
            if ret_val.replan:
                return ret_val
        return MiniSpecReturnValue.default()

    def execute_conditional(self, statement) -> MiniSpecReturnValue:
        condition, program = statement.split('{', 1)
        condition = condition[1:].strip()
        program = program[:-1]

        condition = self.evaluate_condition(condition)
        if condition.replan:
            return condition
        if condition.value:
            return self.execute(program)
        return MiniSpecReturnValue.default()

    def execute_function_call(self, statement) -> MiniSpecReturnValue:
        splits = statement.split('=', 1)
        split_count = len(splits)
        if split_count == 2:
            var, func = splits
            ret_val = self.call_function(func)
            if not ret_val.replan:
                self.env[var.strip()] = ret_val.value
            return ret_val
        elif split_count == 1:
            return self.call_function(statement)
        else:
            raise Exception('Invalid function call statement')

    def evaluate_condition(self, condition) -> MiniSpecReturnValue:
        if '&' in condition:
            conditions = condition.split('&')
            cond = True
            for c in conditions:
                ret_val = self.evaluate_condition(c)
                if ret_val.replan:
                    return ret_val
                cond = cond and ret_val.value
            return MiniSpecReturnValue(cond, False)
        if '|' in condition:
            conditions = condition.split('|')
            for c in conditions:
                ret_val = self.evaluate_condition(c)
                if ret_val.replan:
                    return ret_val
                if ret_val.value == True:
                    return MiniSpecReturnValue(True, False)
            return MiniSpecReturnValue(False, False)
        
        operand_1, comparator, operand_2 = re.split(r'(>|<|==|!=)', condition)

        operand_1 = self.evaluate_operand(operand_1)
        if operand_1.replan:
            return operand_1
        operand_2 = self.evaluate_operand(operand_2)
        if operand_2.replan:
            return operand_2
        
        if type(operand_1.value) != type(operand_2.value):
            raise Exception('Invalid comparison, type mismatch')
        
        if comparator == '>':
            cmp = operand_1.value > operand_2.value
        elif comparator == '<':
            cmp = operand_1.value < operand_2.value
        elif comparator == '==':
            cmp = operand_1.value == operand_2.value
        elif comparator == '!=':
            cmp = operand_1.value != operand_2.value
        else:
            raise Exception(f'Invalid comparator: {comparator}')
        
        return MiniSpecReturnValue(cmp, False)
        
    def evaluate_operand(self, operand) -> MiniSpecReturnValue:
        operand = operand.strip()
        if len(operand) == 0:
            raise Exception('Empty operand')
        if operand.startswith('_'):
            # variable
            return MiniSpecReturnValue(self.get_env_value(operand), False)
        elif operand == 'True' or operand == 'False':
            # boolean
            return MiniSpecReturnValue(evaluate_value(operand), False)
        elif operand[0].isalpha():
            # function call
            return self.execute_function_call(operand)
        else:
            # value
            return MiniSpecReturnValue(evaluate_value(operand), False)

    def call_function(self, func) -> MiniSpecReturnValue:
        splits = func.split('(', 1)
        name = splits[0].strip()
        if len(splits) == 2:
            args = splits[1].strip()[:-1]
            args = split_args(args)
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip('\'"')
                if args[i].startswith('_'):
                    args[i] = self.get_env_value(args[i])
        else:
            args = []
        print(f'Calling skill {name} with args {args}')
        skill_instance = None
        try:
            skill_instance = MiniSpecInterpreter.low_level_skillset.get_skill_by_abbr(name)
        except:
            pass
        if skill_instance is not None:
            return MiniSpecReturnValue.from_tuple(skill_instance.execute(args))

        skill_instance = MiniSpecInterpreter.high_level_skillset.get_skill_by_abbr(name)
        if skill_instance is not None:
            interpreter = MiniSpecInterpreter()
            val = interpreter.execute(skill_instance.execute(args))
            if val.value == 'Replan':
                return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
            return val
        raise Exception(f'Skill {name} is not defined')
    
    '''
    Syntax checking
    '''
    def check_syntax(self, code):
        self.assigned_variables = set()
        return self.check_statement(code)

    def check_statement(self, code):
        """
        Checks the syntax of the given code.
        """
        message = []
        if code.count('{') != code.count('}'):
            message.append('Syntax error: unbalanced brackets')
            return message

        statements = self.split_statements(code)
        for statement in statements:
            if statement.startswith('->'):
                message = message + self.check_return_syntax(statement)
            elif statement[1:].lstrip().startswith('{'):
                message = message + self.check_loop_syntax(statement)
            elif statement.startswith('?'):
                message = message + self.check_conditional_syntax(statement)
            else:
                message = message + self.check_function_call_syntax(statement)

        return message

    def check_return_syntax(self, statement):
        """
        Checks the syntax of a return statement.
        """
        # Add logic to check return statement syntax if needed
        return []

    def check_loop_syntax(self, statement):
        """
        Checks the syntax of a loop statement.
        """
        message = []
        # Extract loop count and program
        count, program = re.match(r'(\d+)\s*\{(.+)\}', statement).groups()
        if not count.isdigit() or int(count) <= 0:
            message.append(f'Invalid loop count: {count}')
        message = message + self.check_statement(program)
        return message

    def check_conditional_syntax(self, statement):
        """
        Checks the syntax of a conditional statement.
        """
        message = []
        # print(statement)
        condition, program = statement.split('{', 1)
        condition = condition[1:].strip()
        program = program[:-1]
        condition = re.split(r'[&|]', condition)
        for cond in condition:
            match_res = re.match(r'(_\d+)\s*(==|!=|<|>)\s*(.+)', cond)
            if not match_res:
                message.append(f'Invalid condition: {cond}')
                continue
            var, comparator, value = match_res.groups()
            if not var.startswith('_'):
                message.append(f'Invalid variable name: {var}')
            if var not in self.assigned_variables:
                message.append(f'Variable {var} is used before assignment')

            if comparator not in ['==', '!=', '<', '>']:
                message.append(f'Invalid comparator: {comparator}')
        # Add logic to check condition syntax
        message = message + self.check_statement(program)
        return message

    def check_function_call_syntax(self, statement):
        """
        Checks the syntax of a function call.
        """
        message = []
        if '=' in statement:
            var, func = statement.split('=')
            var = var.strip()
            if not var.startswith('_'):
                message.append(f'Invalid variable name: {var}')
            self.assigned_variables.add(var)
            message = message + self.check_function_call(func)
        else:
            message = message + self.check_function_call(statement)
        return message

    def check_function_call(self, func):
        """
        Check a function call and adds it to the function_calls list.
        """
        message = []
        match_res = re.match(r'(\w+)(?:,(.+))?', func)
        if not match_res:
            message.append(f'Invalid function call: {func}')
            return message
        name, args = match_res.groups()
        # print(f'Checking syntax for skill {name} with args {args}')
        if args:
            args = [segment for segment in re.findall(r'(?:["\'].*?["\']|[^",]*)(?:,|$)', args) if segment]
            # Check if variables used in args are assigned
            for arg in args:
                if arg.startswith('_') and arg not in self.assigned_variables:
                    raise Exception(f'Variable {arg} is used before assignment')
        else:
            args = []
            
        # print(f'Checking syntax for skill {name} with args {args}')

        # Add logic to check function call syntax
        skill_instance = None
        try:
            skill_instance = MiniSpecInterpreter.low_level_skillset.get_skill_by_abbr(name)
        except:
            try:
                skill_instance = MiniSpecInterpreter.high_level_skillset.get_skill_by_abbr(name)
            except:
                message.append(f'Skill {name} is not defined')

        if skill_instance is not None:
            try:
                skill_instance.parse_args(args, allow_positional_args=True)
            except Exception as e:
                message = message + [str(e)]
        return message