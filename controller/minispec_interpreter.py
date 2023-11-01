import re

class MiniSpecInterpreter:
    def __init__(self):
        self.env = {}
        pass

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
                    start = i+1
            elif char == ';' and len(stack) == 0:
                statements.append(code[start:i].strip())
                start = i+1

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
                self.execute_loop(statement)
            elif '?' in statement:
                self.execute_conditional(statement)
            else:
                self.execute_function_call(statement)

    def evaluate_return(self, statement):
        _, value = statement.split('->')
        return self.evaluate_value(value.strip())
    
    def execute_loop(self, statement):
        count, program = re.match(r'(\d+)\s*\{(.+)\}', statement).groups()
        for i in range(int(count)):
            print(f'Executing loop iteration {i}')
            self.execute(program)

    def execute_conditional(self, statement):
        condition, program = statement.split('{')
        condition = condition[1:].strip()
        program = program[:-1]
        if self.evaluate_condition(condition):
            self.execute(program)

    def execute_function_call(self, statement):
        if '=' in statement:
            var, func = statement.split('=')
            self.env[var.strip()] = self.call_function(func)
        else:
            self.call_function(statement)

    def evaluate_condition(self, condition):
        if '&' in condition:
            conditions = condition.split('&')
            return all(map(self.evaluate_condition, conditions))
        if '|' in condition:
            conditions = condition.split('|')
            return any(map(self.evaluate_condition, conditions))
        var, comparator, value = re.match(r'(_\d+)\s*(==|!=|<|>)\s*(.+)', condition).groups()
        if comparator == '>':
            return self.env[var] > self.evaluate_value(value)
        elif comparator == '<':
            return self.env[var] < self.evaluate_value(value)
        elif comparator == '==':
            return self.env[var] == self.evaluate_value(value)
        elif comparator == '!=':
            return self.env[var] != self.evaluate_value(value)

    def call_function(self, func):
        name, args = re.match(r'(\w+)(?:,(.+))?', func).groups()
        if args:
            args = [self.evaluate_value(arg) for arg in args.split(',')]
        print(f'Calling function {name} with args {args}')
        return 0.1

    def evaluate_value(self, value):
        if value.startswith('_'):
            return self.env[value]
        elif value.isdigit():
            return int(value)
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        elif value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            return value.strip('\'"')

if __name__ == '__main__':
    minispec = MiniSpecInterpreter()
    minispec.execute("mk;4{_2=oy,$1;?_2>0.6{md,20;ap};?_2<0.4{mu,'leo na';ap};_3=oy,$1;?_3<0.6&_3>0.4{->True}};->False")