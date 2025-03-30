from dataclasses import dataclass
import re, queue
from enum import Enum, auto
import time
from threading import Thread
from openai import Stream

from .robot_wrapper import RobotWrapper
from .robot_info import RobotInfo
from .skill_item import SKILL_RET_TYPE
from .utils import print_t, evaluate_value

def _print_debug(*args):
    print(*args)
    # pass

@dataclass
class MiniSpecReturnValue:
    value: SKILL_RET_TYPE
    replan: bool

    @classmethod
    def from_tuple(cls, t: tuple[SKILL_RET_TYPE, bool]) -> 'MiniSpecReturnValue':
        return cls(t[0], t[1])
    
    @classmethod
    def default(cls) -> 'MiniSpecReturnValue':
        return cls(None, False)
    
    def __repr__(self) -> str:
        return f'value={self.value}, replan={self.replan}'

LLM_PLAN_START_PREFIX = '<plan,'
class ProgramParsingState(Enum):
    NONE = auto()
    JSON_BEGIN = auto()
    PREFIX = auto()
    ROBOT_ID = auto()
    QUOTATION_START = auto()
    PLAN = auto()
    QUOTATION_END = auto()
    JSON_END = auto()

class MiniSpecProgram:
    def __init__(self, robot_dict: dict[RobotInfo, RobotWrapper], message_queue: queue.Queue=None) -> None:
        self.parse_state: ProgramParsingState = ProgramParsingState.NONE
        self.parse_buffer: str = ''
        self.skip: int = 0

        self.statement = None
        self.robot_dict = robot_dict
        self.selected_robot = None
        self.message_queue = message_queue

    def parse(self, json_output: Stream | str, stream_interpreting: bool=False) -> bool:
        for chunk in json_output:
            # Get the code from the chunk
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            
            # Skip empty code
            if code == None or len(code) == 0:
                continue

            for c in code:
                if self.skip > 0:
                    self.skip -= 1
                    continue
                match self.parse_state:
                    case ProgramParsingState.NONE:
                        if c == '{':
                            self.parse_state = ProgramParsingState.JSON_BEGIN

                    case ProgramParsingState.JSON_BEGIN:
                        if c == '<':
                            self.parse_buffer = c
                            self.parse_state = ProgramParsingState.PREFIX

                    # match for LLM_PLAN_START_PREFIX
                    case ProgramParsingState.PREFIX:
                        self.parse_buffer += c
                        if self.parse_buffer == LLM_PLAN_START_PREFIX:
                            self.parse_buffer = ''
                            self.parse_state = ProgramParsingState.ROBOT_ID
                            self.skip = 1
                        elif not LLM_PLAN_START_PREFIX.startswith(self.parse_buffer):
                            self.parse_state = ProgramParsingState.JSON_BEGIN

                    case ProgramParsingState.ROBOT_ID:
                        if c == '>':
                            self.parse_state = ProgramParsingState.QUOTATION_START
                            # match the id with the robot list
                            for robot_info, robot in self.robot_dict.items():
                                if robot_info.robot_id == self.parse_buffer:
                                    self.selected_robot = robot
                                    break
                            if not self.selected_robot:
                                raise Exception(f'Invalid robot id: {self.parse_buffer}')
                        else:
                            self.parse_buffer += c

                    case ProgramParsingState.QUOTATION_START:
                        if c == ':':
                            self.parse_buffer = c
                        elif self.parse_buffer == ':' and c == '"':
                            self.parse_buffer = ''
                            self.parse_state = ProgramParsingState.PLAN
                            self.statement = Statement({}, self.selected_robot)
                            self.statement.parse('{')
                        else:
                            continue

                    case ProgramParsingState.PLAN:
                        if c == '"':
                            self.parse_state = ProgramParsingState.QUOTATION_END
                            if self.statement.parse('}'):
                                # print(self.statement.to_string())
                                return True
                        else:
                            if self.message_queue:
                                self.message_queue.put(c + '\\\\')

                            self.statement.parse(c, stream_interpreting)
                            # # TODO: test executable
                            # if stream_interpreting and self.statement.executable and not self.statement.running:
                            #     # Send the statement to the execution queue
                            #     print(f'##### Statement executable: {self.statement.action}')
                            #     self.statement.running = True
                            #     Statement.execution_queue.put(self.statement)

                    case ProgramParsingState.QUOTATION_END:
                        if c == '}':
                            self.parse_state = ProgramParsingState.JSON_END
                    
                    case ProgramParsingState.JSON_END:
                        return False
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        return self.statement.eval()
    
class CodeAction(Enum):
    NONE = auto()
    ATOMIC = auto() # single statement
    SEQ = auto()    # sequence of statements
    IF = auto()     # if statement
    LOOP = auto()   # loop statement

class StatementParsingState(Enum):
    DEFAULT = auto()
    ARGUMENTS = auto()
    CONDITION = auto()
    LOOP_COUNT = auto()
    IF_SUB_STATEMENT = auto()
    ELSE_SUB_STATEMENT = auto()

class Statement:
    def __init__(self, env: dict, robot: RobotWrapper):
        self.parse_state = StatementParsingState.DEFAULT
        self.parse_buffer: str = ''
        self.parse_depth: int = 0

        self.action = CodeAction.NONE
        self.condition: list[str] = [] # `if`, `elif`, ...
        self.loop_count: int = 0
        self.current_statement = None

        # ATOMIC: single statement
        # SEQ: list of statements
        # IF: list of statements corresponding to conditions.
        #     The last statement is the `else` statement if len(sub_statements) > len(condition).
        # LOOP: single statement
        self.sub_statements: list[str | 'Statement'] = []
        
        self.allow_digit: bool = False
        self.quotation: bool = False

        self.executable: bool = False

        self.ret: bool = False
        self.env = env
        self.robot = robot

    """
    Print the statement in a simple format for debugging.
    """
    def to_string_simple(self) -> str:
        s = ''
        if self.action == CodeAction.IF:
            s += f'IF {self.condition[0]} {{...}}'
        elif self.action == CodeAction.LOOP:
            s += f'[{self.loop_count}] {{...}}'
        elif self.action == CodeAction.SEQ:
            s += '{...}'
        elif self.action == CodeAction.ATOMIC:
            s += f'{self.sub_statements[0]}'
        else:
            raise Exception('Invalid action')
        
        return s

    """
    Print the full statement in a human-readable format for debugging.
    """
    def to_string(self, depth: int=0) -> str:
        indent = '****'
        prefix = indent * depth
        s = ''
        if self.action == CodeAction.IF:
            len1 = len(self.condition)
            len2 = len(self.sub_statements)
            
            for i in range(len1):
                if i > 0:
                    s += ' else '
                else:
                    s += prefix
                s += f'if {self.condition[i]}\n'
                s += f'{self.sub_statements[i].to_string(depth)}'

            if len2 > len1:
                s += ' else\n'
                s += self.sub_statements[-1].to_string(depth)

        elif self.action == CodeAction.LOOP:
            s += prefix + f'[{self.loop_count}]\n'
            s += self.sub_statements[0].to_string(depth)
        elif self.action == CodeAction.SEQ:
            s += prefix + '{\n'
            for statement in self.sub_statements:
                s += f'{statement.to_string(depth + 1)};\n'
            s += prefix + '}'
        elif self.action == CodeAction.ATOMIC:
            if not isinstance(self.sub_statements[0], str):
                raise Exception('Invalid action')
            s += prefix + self.sub_statements[0]
        else:
            raise Exception('Invalid action')
        
        return s
    
    def _get_env(self, var) -> SKILL_RET_TYPE:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        for c in code:
            if c == ' ' and not self.quotation:
                continue
            # print('--' * self.depth + f'-> {c}, action: {self.action}, state: {self.parse_state}')

            if c == '\'':
                self.quotation = not self.quotation

            match self.action:
                case CodeAction.NONE:
                    if c == '{':
                        self.action = CodeAction.SEQ
                        self.current_statement = Statement(self.env, self.robot)
                        self.parse_depth += 1
                    elif c == '?':
                        self.action = CodeAction.IF
                        self.parse_buffer = ''
                        self.parse_state = StatementParsingState.CONDITION
                    elif c == ';' or c == '}':
                        return False
                    elif c.isalpha() or c == '_' or c == '-':
                        self.parse_buffer = c
                        self.action = CodeAction.ATOMIC
                        self.allow_digit = True
                    elif c.isdigit() and not self.allow_digit:
                        self.action = CodeAction.LOOP
                        self.parse_state = StatementParsingState.LOOP_COUNT
                        self.parse_buffer += c
                    else:
                        raise Exception(f'Invalid character: {c}')

                case CodeAction.ATOMIC:
                    if not self.quotation and (c == ';' or c == '}'):
                        self.sub_statements.append(self.parse_buffer)
                        self.executable = True
                        return True
                    else:
                        self.parse_buffer += c

                case CodeAction.SEQ:
                    done = self.current_statement.parse(c)

                    if self.current_statement.executable:
                        self.executable = True

                    if done:
                        self.sub_statements.append(self.current_statement)
                        self.current_statement = Statement(self.env, self.robot)
                        self.current_statement.parse(c)

                    if self.quotation:
                        continue

                    if c == '{':
                        self.parse_depth += 1
                    elif c == '}':
                        self.parse_depth -= 1
                        if self.parse_depth == 0:
                            return True

                case CodeAction.IF:
                    match self.parse_state:
                        case StatementParsingState.DEFAULT:
                            if c != ':':
                                return True
                            else:
                                self.parse_state = StatementParsingState.ELSE_SUB_STATEMENT
                        case StatementParsingState.CONDITION:
                            if c == '{' and not self.quotation:
                                self.condition.append(self.parse_buffer)
                                self.executable = True
                                self.parse_state = StatementParsingState.IF_SUB_STATEMENT
                                self.current_statement = Statement(self.env, self.robot)
                                self.current_statement.parse(c)
                                self.parse_depth += 1
                            else:
                                # read condition between '?' and '{'
                                self.parse_buffer += c
                        case StatementParsingState.IF_SUB_STATEMENT:
                            if self.current_statement.parse(c):
                                self.sub_statements.append(self.current_statement)
                                self.current_statement = Statement(self.env, self.robot)

                            if c == '{':
                                self.parse_depth += 1
                            elif c == '}':
                                self.parse_depth -= 1
                                if self.parse_depth == 0:
                                    self.parse_state = StatementParsingState.DEFAULT
                        case StatementParsingState.ELSE_SUB_STATEMENT:
                            if c == '?':
                                self.parse_buffer = ''
                                self.parse_state = StatementParsingState.CONDITION
                            elif c == '{':
                                self.current_statement = Statement(self.env, self.robot)
                                self.current_statement.parse(c)
                                self.parse_state = StatementParsingState.IF_SUB_STATEMENT
                                self.parse_depth += 1
                            else:
                                raise Exception(f'Invalid character: {c}')

                case CodeAction.LOOP:
                    match self.parse_state:
                        case StatementParsingState.LOOP_COUNT:
                            if c == '{':
                                self.loop_count = int(self.parse_buffer)
                                self.parse_buffer = ''
                                self.parse_state = StatementParsingState.DEFAULT
                                self.current_statement = Statement(self.env, self.robot)
                                self.current_statement.parse(c)
                            elif c.isdigit():
                                self.parse_buffer += c
                            else:
                                raise Exception(f'Invalid loop count: {self.parse_buffer}')
                        case StatementParsingState.DEFAULT:
                            done = self.current_statement.parse(c)
                            if self.current_statement.executable:
                                self.executable = True

                            if done:
                                self.sub_statements.append(self.current_statement)
                                return True
                            
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        _print_debug(f'Eval statement: {self.to_string_simple()}')
        while not self.executable:
            time.sleep(0.1)
        default_ret_val = MiniSpecReturnValue.default()

        match self.action:
            case CodeAction.ATOMIC:
                assert len(self.sub_statements) == 1 and isinstance(self.sub_statements[0], str)
                return self.eval_expr(self.sub_statements[0])

            case CodeAction.SEQ:
                for statement in self.sub_statements:
                    ret_val = statement.eval()
                    if ret_val.replan or statement.ret:
                        self.ret = True
                        return ret_val

            case CodeAction.IF:
                for i in range(len(self.condition)):
                    condition_val = self.eval_condition(self.condition[i])
                    if condition_val.replan:
                        return condition_val
                    
                    if condition_val.value == True:
                        ret_val = self.sub_statements[i].eval()
                        if self.sub_statements[i].ret:
                            self.ret = True
                        return ret_val
                
                if len(self.sub_statements) > len(self.condition):
                    ret_val = self.sub_statements[-1].eval()
                    if self.sub_statements[-1].ret:
                        self.ret = True
                    return ret_val

            case CodeAction.LOOP:
                assert len(self.sub_statements) == 1
                for i in range(self.loop_count):
                    ret_val = self.sub_statements[0].eval()
                    if ret_val.replan or self.sub_statements[0].ret:
                        self.ret = True
                        return ret_val

            case CodeAction.NONE:
                raise Exception('Invalid action')
        return default_ret_val

    def eval_function(self, func: str) -> MiniSpecReturnValue:
        _print_debug(f'Eval function: {func}')

        lp = func.find('(')
        rp = func.rfind(')')
        if lp == -1 or rp == -1 or lp > rp:
            raise Exception(f'Invalid function call: {func}')

        func_name = func[:lp].strip()
        args = func[lp + 1:rp].strip()

        if len(args) == 0:
            args = []
        else:
            args = re.split(r",\s*(?![^()]*\))", args)

        for i in range(len(args)):
            val = self.eval_expr(args[i])
            if val.replan:
                return val
            args[i] = val.value
            
        print(func_name, args)

        ll_skill = self.robot.ll_skillset.get_skill(func_name)
        if ll_skill:
            rslt = ll_skill.execute(args)
            _print_debug(f'Executing low-level skill: {ll_skill.name} {args} {rslt}')
            return MiniSpecReturnValue.from_tuple(rslt)

        hl_skill = self.robot.hl_skillset.get_skill(func_name)
        if hl_skill:
            _print_debug(f'Executing high-level skill: {hl_skill.name}', args, hl_skill.execute(args)[0])
            s = Statement(self.env, self.robot)
            s.parse(hl_skill.execute(args)[0])
            val = s.eval()
            return val
        
        raise Exception(f'Skill {func_name} is not defined')

    def eval_expr(self, expr: str) -> MiniSpecReturnValue:
        _print_debug(f'Eval expr: {expr}')
        expr = expr.strip()

        # stripe ()
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()

        if len(expr) == 0:
            raise Exception('Empty expression')
        
        # Handle return value (->)
        if expr.startswith('->'):
            self.ret = True
            return MiniSpecReturnValue(self.eval_expr(expr.lstrip('->')).value, False)
        
        # Handle variable assignment (_var = ...)
        if expr.startswith('_') and '=' in expr:
            var, expr = expr.split('=', 1)
            var = var.strip()
            _print_debug(f'Eval expr var assign: {var}={expr}')
            ret_val = self.eval_expr(expr)
            _print_debug(f'==> var assign: {var}={ret_val.value}')
            self.env[var] = ret_val.value
            return ret_val
        
        # Handle arithmetic operations
        operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
        }

        # Split the expression while respecting parentheses and negative numbers
        def split_expression(expr, operator):
            depth = 0
            parts = []
            current = []
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == operator and depth == 0:
                    # Ensure we don't split on a negative sign (e.g., "-10")
                    if operator == '-' and (i == 0 or expr[i - 1] in ' ('):
                        current.append(char)
                        continue
                    parts.append(''.join(current).strip())
                    current = []
                    continue
                current.append(char)
            parts.append(''.join(current).strip())
            return parts

        for op, func in operators.items():
            if op in expr:
                operands = split_expression(expr, op)
                print(f'Operands: {operands}')
                if len(operands) < 2:
                    continue
                # Evaluate the first operand
                result = self.eval_expr(operands[0]).value
                # Apply the operator to the remaining operands
                for operand in operands[1:]:
                    result = func(result, self.eval_expr(operand.strip()).value)
                return MiniSpecReturnValue(result, False)

        # Handle variables, constants, and function calls
        if expr.startswith('_'):
            return MiniSpecReturnValue(self._get_env(expr), False)
        elif expr == 'True' or expr == 'False':
            return MiniSpecReturnValue(evaluate_value(expr), False)
        elif expr[0].isalpha():
            return self.eval_function(expr)
        else:
            return MiniSpecReturnValue(evaluate_value(expr), False)

    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        ### TODO: add support for nested conditions

        # Multiple conditions
        if '&&' in condition:
            conditions = condition.split('&&')
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan or ret_val.value == False:
                    return ret_val
            return MiniSpecReturnValue(True, False)
        if '||' in condition:
            conditions = condition.split('||')
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan or ret_val.value != False:
                    return ret_val
            return MiniSpecReturnValue(False, False)
        
        # Single condition
        parts = re.split(r'(>|<|==|!=)', condition)
        if len(parts) != 1 and len(parts) != 3:
            raise Exception(f'Invalid condition format: {condition}')

        operand_1 = parts[0]
        operand_1 = self.eval_expr(operand_1)
        if operand_1.replan:
            return operand_1

        if len(parts) == 3:
            comparator, operand_2 = parts[1], parts[2]
            operand_2 = self.eval_expr(operand_2)
            if operand_2.replan:
                return operand_2
            _print_debug(f'Condition ops: {operand_1.value} {comparator} {operand_2.value}')
        else:
            _print_debug(f'Condition ops: {operand_1.value}')
            return MiniSpecReturnValue(operand_1.value != False, False)

        if isinstance(operand_1.value, (int, float)) and isinstance(operand_2.value, (int, float)):
            operand_1.value = float(operand_1.value)
            operand_2.value = float(operand_2.value)

        if type(operand_1.value) != type(operand_2.value):
            if comparator == '!=':
                return MiniSpecReturnValue(True, False)
            elif comparator == '==':
                return MiniSpecReturnValue(False, False)
            else:
                raise Exception(f'Invalid comparator: {operand_1.value}:{type(operand_1.value)} {operand_2.value}:{type(operand_2.value)}')

        if comparator == '>':
            cmp = operand_1.value > operand_2.value
        elif comparator == '<':
            cmp = operand_1.value < operand_2.value
        elif comparator == '==':
            cmp = operand_1.value == operand_2.value
        elif comparator == '!=':
            cmp = operand_1.value != operand_2.value
        
        return MiniSpecReturnValue(cmp, False)

class MiniSpecInterpreter:
    def __init__(self, message_queue: queue.Queue, robot_dict: dict[RobotInfo, RobotWrapper]):
        self.robot_dict = robot_dict
        self.message_queue = message_queue

        self.execution_history = []

        self.program = None
        self.execution_thread = Thread(target=self.executor)

    def execute(self, json_output: Stream | str) -> MiniSpecReturnValue:
        self.execution_history = []
        self.timestamp_get_plan = time.time()

        stream_interpreting = False if isinstance(json_output, str) else True

        if stream_interpreting:
            self.execution_thread.start()

        self.program = MiniSpecProgram(self.robot_dict, self.message_queue)
        self.program.parse(json_output, stream_interpreting)

        if stream_interpreting:
            print_t(f"[M] Program received in {time.time() - self.timestamp_get_plan}s")
        else:
            print_t("[M] Start normal execution")
            self.program.eval()

    def executor(self):
        while True:
            if self.program and self.program.statement and self.program.statement.executable:
                print_t("[M] Start execution")
                self.program.statement.eval()
                return
            else:
                time.sleep(0.005)
