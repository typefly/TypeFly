from dataclasses import dataclass
from typing import Optional
import re, queue
from enum import Enum, auto
import time
from threading import Thread
from queue import Queue
from openai import Stream
from .skill_item import SKILL_RET_TYPE
from .skillset import SkillSet
from .utils import split_args, print_t

def _print_debug(*args):
    print(*args)
    # pass

def evaluate_value(value: str) -> SKILL_RET_TYPE:
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
    QUOTATION_START = auto()
    PLAN = auto()
    QUOTATION_END = auto()
    JSON_END = auto()

class MiniSpecProgram:
    def __init__(self, env: Optional[dict]=None, message_queue: queue.Queue=None) -> None:
        self.parse_state: ProgramParsingState = ProgramParsingState.NONE
        self.parse_buffer: str = ''
        self.skip: int = 0
  
        self.finished = False
        self.ret = False
        self.env = env or {}

        self.statements: list[Statement] = []
        self.current_statement = None

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
                            self.parse_state = ProgramParsingState.QUOTATION_START
                        elif not LLM_PLAN_START_PREFIX.startswith(self.parse_buffer):
                            self.parse_state = ProgramParsingState.JSON_BEGIN
                    case ProgramParsingState.QUOTATION_START:
                        if c == ':':
                            self.parse_buffer = c
                        elif self.parse_buffer == ':' and c == '"':
                            self.parse_buffer = ''
                            self.parse_state = ProgramParsingState.PLAN
                            self.current_statement = Statement(self.env)
                        else:
                            continue
                    case ProgramParsingState.PLAN:
                        if c == '"':
                            self.parse_state = ProgramParsingState.QUOTATION_END
                        else:
                            # Send the code piece to the message queue
                            if self.message_queue:
                                self.message_queue.put(c + '\\\\')

                            if stream_interpreting and self.current_statement.executable:
                                # Send the statement to the execution queue
                                print(f'Adding statement: {self.current_statement}')

                            if self.current_statement.finished:
                                self.statements.append(self.current_statement)
                                self.current_statement = Statement(self.env)

                            self.current_statement.parse(c, stream_interpreting)

                    case ProgramParsingState.QUOTATION_END:
                        if c == '}':
                            self.parse_state = ProgramParsingState.JSON_END
                    
                    case ProgramParsingState.JSON_END:
                        return True
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        _print_debug(f'Eval program: {self}, finished: {self.finished}')
        ret_val = MiniSpecReturnValue.default()
        count = 0
        while not self.finished:
            if len(self.statements) <= count:
                time.sleep(0.1)
                continue
            ret_val = self.statements[count].eval()
            if ret_val.replan or self.statements[count].ret:
                _print_debug(f'RET from {self.statements[count]} with {ret_val} {self.statements[count].ret}')
                self.ret = True
                return ret_val
            count += 1
        if count < len(self.statements):
            for i in range(count, len(self.statements)):
                ret_val = self.statements[i].eval()
                if ret_val.replan or self.statements[i].ret:
                    _print_debug(f'RET from {self.statements[i]} with {ret_val} {self.statements[i].ret}')
                    self.ret = True
                    return ret_val
        return ret_val
    
    def __repr__(self) -> str:
        s = ''
        for statement in self.statements:
            s += f'{statement}; '
        return s
    
class CodeAction(Enum):
    NONE = auto()
    ATOMIC = auto()
    SEQ = auto()
    IF = auto()
    LOOP = auto()

class StatementParsingState(Enum):
    DEFAULT = auto()
    ARGUMENTS = auto()
    CONDITION = auto()
    LOOP_COUNT = auto()
    IF_SUB_STATEMENT = auto()
    ELSE_SUB_STATEMENT = auto()

class Statement:
    def __init__(self, env: dict):
        self.parse_state = StatementParsingState.DEFAULT
        self.parse_buffer: str = ''
        self.parse_depth: int = 0

        self.action = CodeAction.NONE
        self.condition: list[str] = []
        self.loop_count: int = 0
        self.current_statement = None
        self.sub_statements: list[str | 'Statement'] = []
        
        self.allow_digit: bool = False
        self.quotation: bool = False

        self.executable: bool = False
        self.finished: bool = False

        self.ret: bool = False
        self.env = env

    def to_string(self, depth: int=0) -> str:
        indent = '_-_-'
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
                        self.current_statement = Statement(self.env)
                        self.parse_depth += 1
                    elif c == '?':
                        self.action = CodeAction.IF
                        self.parse_buffer = ''
                        self.parse_state = StatementParsingState.CONDITION
                    elif c == ';' or c == '}':
                        return False
                    elif c.isalpha() or c == '_':
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
                        self.finished = True
                        return True
                    else:
                        self.parse_buffer += c

                case CodeAction.SEQ:
                    if self.current_statement.parse(c):
                        self.sub_statements.append(self.current_statement)
                        self.current_statement = Statement(self.env)

                    if self.current_statement.executable:
                        self.executable = True

                    if self.quotation:
                        continue

                    if c == '{':
                        self.parse_depth += 1
                    elif c == '}':
                        self.parse_depth -= 1
                        if self.parse_depth == 0:
                            self.finished = True
                            return True

                case CodeAction.IF:
                    match self.parse_state:
                        case StatementParsingState.DEFAULT:
                            if c != ':':
                                self.finished = True
                                return True
                            else:
                                self.parse_state = StatementParsingState.ELSE_SUB_STATEMENT
                        case StatementParsingState.CONDITION:
                            if c == '{' and not self.quotation:
                                self.condition.append(self.parse_buffer)
                                self.executable = True
                                self.parse_state = StatementParsingState.IF_SUB_STATEMENT
                                self.current_statement = Statement(self.env)
                                self.current_statement.parse(c)
                                self.parse_depth += 1
                            else:
                                # read condition between '?' and '{'
                                self.parse_buffer += c
                        case StatementParsingState.IF_SUB_STATEMENT:
                            if self.current_statement.parse(c):
                                self.sub_statements.append(self.current_statement)
                                self.current_statement = Statement(self.env)

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
                                self.current_statement = Statement(self.env)
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
                                self.current_statement = Statement(self.env)
                                self.current_statement.parse(c)
                            elif c.isdigit():
                                self.parse_buffer += c
                            else:
                                raise Exception(f'Invalid loop count: {self.parse_buffer}')
                        case StatementParsingState.DEFAULT:
                            if self.current_statement.parse(c):
                                self.sub_statements.append(self.current_statement)
                                self.finished = True
                                return True
                            
                            if self.current_statement.executable:
                                self.executable = True
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        _print_debug(f'Statement eval: {self} {self.action} {self.condition} {self.loop_count}')
        while not self.executable:
            time.sleep(0.1)
        if self.action == 'if':
            ret_val = self.eval_condition(self.condition)
            if ret_val.replan:
                return ret_val
            if ret_val.value:
                _print_debug(f'-> eval condition statement: {self.sub_statements}')
                ret_val = self.sub_statements.eval()
                if ret_val.replan or self.sub_statements.ret:
                    self.ret = True
                return ret_val
            else:
                return MiniSpecReturnValue.default()
        elif self.action == 'loop':
            _print_debug(f'-> eval loop statement: {self.loop_count} {self.sub_statements}')
            ret_val = MiniSpecReturnValue.default()
            for _ in range(self.loop_count):
                _print_debug(f'-> loop iteration: {ret_val}')
                ret_val = self.sub_statements.eval()
                if ret_val.replan or self.sub_statements.ret:
                    self.ret = True
                    return ret_val
            return ret_val
        else:
            self.ret = False
            return self.eval_expr(self.action)
    
    # def eval_action(self, action: str) -> MiniSpecReturnValue:
    #     action = action.strip()
    #     _print_debug(f'Eval action: {action}')
        
    #     if '=' in action:
    #         var, expr = action.split('=')
    #         _print_debug(f'Assignment: Var: {var.strip()}, Val: {expr.strip()}')
    #         expr = expr.strip()
    #         ret_val = self.eval_function(expr.strip())
    #         if not ret_val.replan:
    #             self.env[var.strip()] = ret_val.value
    #         return ret_val
    #     elif action.startswith('->'):
    #         self.ret = True
    #         return self.eval_expr(action.lstrip("->"))
    #     else:
    #         return self.eval_function(action)

    def eval_function(self, func: str) -> MiniSpecReturnValue:
        _print_debug(f'Eval function: {func}')
        # append to execution state queue
        func = func.split('(', 1)
        name = func[0].strip()
        if len(func) == 2:
            args = func[1].strip()[:-1]
            args = split_args(args)
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip('\'"')
                if args[i].startswith('_'):
                    args[i] = self._get_env(args[i])
        else:
            args = []

        skill_instance = Statement.low_level_skillset.get_skill(name)
        if skill_instance is not None:
            _print_debug(f'Executing low-level skill: {skill_instance.get_name()} {args}')
            return MiniSpecReturnValue.from_tuple(skill_instance.execute(args))

        skill_instance = Statement.high_level_skillset.get_skill(name)
        if skill_instance is not None:
            _print_debug(f'Executing high-level skill: {skill_instance.get_name()}', args, skill_instance.execute(args))
            interpreter = MiniSpecProgram()
            interpreter.parse([skill_instance.execute(args)])
            interpreter.finished = True
            val = interpreter.eval()
            if val.value == 'rp':
                return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
            return val
        raise Exception(f'Skill {name} is not defined')

    def eval_expr(self, expr: str) -> MiniSpecReturnValue:
        print_t(f'Eval expr: {expr}')
        expr = expr.strip()
        if len(expr) == 0:
            raise Exception('Empty operand')
        
        # Handle return value (->)
        if expr.startswith('->'):
            self.ret = True
            return MiniSpecReturnValue(self.eval_expr(expr.lstrip('->')).value, True)
        
        # Handle variable assignment (_var = ...)
        if expr.startswith('_') and '=' in expr:
            var, expr = expr.split('=', 1)
            var = var.strip()
            print_t(f'Eval expr var assign: {var} {expr}')
            ret_val = self.eval_expr(expr)
            self.env[var] = ret_val.value
            return ret_val
        
        # Handle arithmetic operations
        operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
        }

        for op, func in operators.items():
            if op in expr:
                operands = expr.split(op)
                if len(operands) < 2:
                    raise Exception(f'Invalid expression: {expr}')
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
            return MiniSpecReturnValue(operand_1.value, False)

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

    def __repr__(self) -> str:
        s = ''
        if self.action == 'if':
            s += f'if {self.condition}'
        elif self.action == 'loop':
            s += f'[{self.loop_count}]'
        else:
            s += f'{self.action}'
        if self.sub_statements is not None:
            s += ' {'
            for statement in self.sub_statements.statements:
                s += f'{statement}; '
            s += '}'
        return s

class MiniSpecInterpreter:
    def __init__(self, message_queue: queue.Queue):
        self.env = {}
        self.ret = False
        self.code_buffer: str = ''

        self.execution_history = []
        # if Statement.low_level_skillset is None or \
        #     Statement.high_level_skillset is None:
        #     raise Exception('Statement: Skillset is not initialized')
        
        Statement.execution_queue = Queue()
        self.execution_thread = Thread(target=self.executor)
        self.execution_thread.start()

        self.timestamp_get_plan = None
        self.timestamp_start_execution = None
        self.timestamp_end_execution = None
        self.program_count = 0
        self.ret_queue = Queue()
        self.message_queue = message_queue

    def execute(self, json_output: Stream | str) -> MiniSpecReturnValue:
        self.execution_history = []
        self.timestamp_get_plan = time.time()

        stream_interpreting = False if isinstance(json_output, str) else True

        program = MiniSpecProgram(message_queue=self.message_queue)
        program.parse(json_output, stream_interpreting)
        self.program_count = len(program.statements)

        if stream_interpreting:
            print_t(f"[M] Program received in {time.time() - self.timestamp_get_plan}s")
        else:
            print_t("[M] Start normal execution")
            program.eval()

    ### TODO: fix this
    def executor(self):
        while True:
            if not Statement.execution_queue.empty():
                if self.timestamp_start_execution is None:
                    self.timestamp_start_execution = time.time()
                    print_t(">>> Start execution")
                statement = Statement.execution_queue.get()
                _print_debug(f'Queue get statement: {statement}')
                ret_val = statement.eval()
                print_t(f'Queue statement done: {statement}')
                if statement.ret:
                    while not Statement.execution_queue.empty():
                        Statement.execution_queue.get()
                    self.ret_queue.put(ret_val)
                    return
                self.execution_history.append(statement)
                # if ret_val.replan:
                #     print_t(f'Queue statement replan: {statement}')
                #     self.ret_queue.put(ret_val)
                #     return
                self.program_count -= 1
                if self.program_count == 0:
                    self.timestamp_end_execution = time.time()
                    print_t(f'>>> Execution time: {self.timestamp_end_execution - self.timestamp_start_execution}')
                    self.timestamp_start_execution = None
                    self.ret_queue.put(ret_val)
                    return
            else:
                time.sleep(0.005)
