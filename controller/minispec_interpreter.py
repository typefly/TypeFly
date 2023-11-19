import re
from .skillset import SkillSet
from .utils import evaluate_value

class MiniSpecInterpreter:
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self):
        self.env = {}
        if MiniSpecInterpreter.low_level_skillset is None or MiniSpecInterpreter.high_level_skillset is None:
            raise Exception('MiniSpecInterpreter: Skillset is not initialized')

    def get_env_value(self, var):
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]
    
    def check_statement(self, code):
        message = []
        if code.count('{') != code.count('}'):
            message.append('Syntax error: unbalanced brackets')

        statements = self.split_statements(code)
        for statement in statements:
            message = message + self.check_statement(statement)

    def split_statements(self, code):
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

    def execute(self, code):
        statements = self.split_statements(code)
        for statement in statements:
            print(f'Executing statement: {statement}')
            if not statement:
                continue
            if statement.startswith('->'):
                return self.evaluate_return(statement)
            elif statement[1:].lstrip().startswith('{'):
                result = self.execute_loop(statement)
                if result is not None:
                    return result
            elif statement.startswith('?'):
                result = self.execute_conditional(statement)
                if result is not None:
                    return result
            else:
                self.execute_function_call(statement)

    def evaluate_return(self, statement):
        _, value = statement.split('->')
        return evaluate_value(value.strip())
    
    def execute_loop(self, statement):
        count, program = re.match(r'(\d+)\s*\{(.+)\}', statement).groups()
        for i in range(int(count)):
            print(f'Executing loop iteration {i}')
            result = self.execute(program)
            if result is not None:
                return result

    def execute_conditional(self, statement):
        condition, program = statement.split('{')
        condition = condition[1:].strip()
        program = program[:-1]
        if self.evaluate_condition(condition):
            return self.execute(program)

    def execute_function_call(self, statement):
        if '=' in statement:
            var, func = statement.split('=')
            self.env[var.strip()] = self.call_function(func)
        else:
            self.call_function(statement)

    def evaluate_condition(self, condition) -> bool:
        if '&' in condition:
            conditions = condition.split('&')
            return all(map(self.evaluate_condition, conditions))
        if '|' in condition:
            conditions = condition.split('|')
            return any(map(self.evaluate_condition, conditions))
        var, comparator, value = re.match(r'(_\d+)\s*(==|!=|<|>)\s*(.+)', condition).groups()
        var_value = self.get_env_value(var)
        if comparator == '>':
            return var_value > evaluate_value(value)
        elif comparator == '<':
            return var_value < evaluate_value(value)
        elif comparator == '==':
            return var_value == evaluate_value(value)
        elif comparator == '!=':
            return var_value != evaluate_value(value)

    def call_function(self, func):
        name, args = re.match(r'(\w+)(?:,(.+))?', func).groups()
        if args:
            args = [segment for segment in re.findall(r'(?:["\'].*?["\']|[^",]*)(?:,|$)', args) if segment]
            # args = args.split(',')
            # replace _1, _2, ... with their values
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip("'")
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
            return skill_instance.execute(args)

        skill_instance = MiniSpecInterpreter.high_level_skillset.get_skill_by_abbr(name)
        if skill_instance is not None:
            interpreter = MiniSpecInterpreter()
            return interpreter.execute(skill_instance.execute(args))
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
        condition, program = statement.split('{')
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
        message = message + self.check_statement(program[:-1])
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