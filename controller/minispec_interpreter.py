from typing import Optional
import re, queue
from enum import Enum
import time
from threading import Thread
from queue import Queue
from openai import ChatCompletion, Stream
from .skill_item import SKILL_RET_TYPE
from .skillset import SkillSet
from .utils import split_args, print_t

def print_debug(*args):
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

class MiniSpecReturnValue:
    def __init__(self, value: SKILL_RET_TYPE, replan: bool):
        self.value = value
        self.replan = replan

    def from_tuple(t: tuple[SKILL_RET_TYPE, bool]):
        return MiniSpecReturnValue(t[0], t[1])
    
    def default():
        return MiniSpecReturnValue(None, False)
    
    def __repr__(self) -> str:
        return f'value={self.value}, replan={self.replan}'
    
class ParsingState(Enum):
    CODE = 0
    ARGUMENTS = 1
    CONDITION = 2
    LOOP_COUNT = 3
    SUB_STATEMENTS = 4

class MiniSpecProgram:
    def __init__(self, env: Optional[dict] = None, mq: queue.Queue = None) -> None:
        self.statements: list[Statement] = []
        self.depth = 0
        self.finished = False
        self.ret = False
        if env is None:
            self.env = {}
        else:
            self.env = env
        self.current_statement = Statement(self.env)
        self.mq = mq

    def parse(self, code_instance: Stream[ChatCompletion.ChatCompletionChunk] | list[str], exec: bool = False) -> bool:
        for chunk in code_instance:
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            if code == None or len(code) == 0:
                continue
            if self.mq:
                self.mq.put(code + '\\\\')
            for c in code:
                if self.current_statement.parse(c, exec):
                    if len(self.current_statement.action) > 0:
                        print_debug("Adding statement: ", self.current_statement, exec)
                        self.statements.append(self.current_statement)
                    self.current_statement = Statement(self.env)
                if c == '{':
                    self.depth += 1
                elif c == '}':
                    if self.depth == 0:
                        self.finished = True
                        return True
                    self.depth -= 1
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Eval program: {self}, finished: {self.finished}')
        ret_val = MiniSpecReturnValue.default()
        count = 0
        while not self.finished:
            if len(self.statements) <= count:
                time.sleep(0.1)
                continue
            ret_val = self.statements[count].eval()
            if ret_val.replan or self.statements[count].ret:
                print_debug(f'RET from {self.statements[count]} with {ret_val} {self.statements[count].ret}')
                self.ret = True
                return ret_val
            count += 1
        if count < len(self.statements):
            for i in range(count, len(self.statements)):
                ret_val = self.statements[i].eval()
                if ret_val.replan or self.statements[i].ret:
                    print_debug(f'RET from {self.statements[i]} with {ret_val} {self.statements[i].ret}')
                    self.ret = True
                    return ret_val
        return ret_val
    
    def __repr__(self) -> str:
        s = ''
        for statement in self.statements:
            s += f'{statement}; '
        return s

class Statement:
    execution_queue: Queue['Statement'] = None
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self, env: dict) -> None:
        self.code_buffer: str = ''
        self.parsing_state: ParsingState = ParsingState.CODE
        self.condition: Optional[str] = None
        self.loop_count: Optional[int] = None
        self.action: str = ''
        self.allow_digit: bool = False
        self.executable: bool = False
        self.ret: bool = False
        self.sub_statements: Optional[MiniSpecProgram] = None
        self.env = env
        self.read_argument: bool = False

    def get_env_value(self, var) -> SKILL_RET_TYPE:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        for c in code:
            match self.parsing_state:
                case ParsingState.CODE:
                    if c == '?' and not self.read_argument:
                        self.action = 'if'
                        self.parsing_state = ParsingState.CONDITION
                    elif c == ';' or c == '}' or c == ')':
                        if c == ')':
                            self.code_buffer += c
                            self.read_argument = False
                        self.action = self.code_buffer
                        print_debug(f'SP Action: {self.code_buffer}')
                        self.executable = True
                        if exec and self.action != '':
                            self.execution_queue.put(self)
                        return True
                    else:
                        if c == '(':
                            self.read_argument = True
                        if c.isalpha() or c == '_':
                            self.allow_digit = True
                        self.code_buffer += c
                    if c.isdigit() and not self.allow_digit:
                        self.action = 'loop'
                        self.parsing_state = ParsingState.LOOP_COUNT
                case ParsingState.CONDITION:
                    if c == '{':
                        print_debug(f'SP Condition: {self.code_buffer}')
                        self.condition = self.code_buffer
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c
                case ParsingState.LOOP_COUNT:
                    if c == '{':
                        print_debug(f'SP Loop: {self.code_buffer}')
                        self.loop_count = int(self.code_buffer)
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c
                case ParsingState.SUB_STATEMENTS:
                    if self.sub_statements.parse([c]):
                        return True
        return False
    
    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Statement eval: {self} {self.action} {self.condition} {self.loop_count}')
        while not self.executable:
            time.sleep(0.1)
        if self.action == 'if':
            ret_val = self.eval_condition(self.condition)
            if ret_val.replan:
                return ret_val
            if ret_val.value:
                print_debug(f'-> eval condition statement: {self.sub_statements}')
                ret_val = self.sub_statements.eval()
                if ret_val.replan or self.sub_statements.ret:
                    self.ret = True
                return ret_val
            else:
                return MiniSpecReturnValue.default()
        elif self.action == 'loop':
            print_debug(f'-> eval loop statement: {self.loop_count} {self.sub_statements}')
            ret_val = MiniSpecReturnValue.default()
            for _ in range(self.loop_count):
                print_debug(f'-> loop iteration: {ret_val}')
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
    #     print_debug(f'Eval action: {action}')
        
    #     if '=' in action:
    #         var, expr = action.split('=')
    #         print_debug(f'Assignment: Var: {var.strip()}, Val: {expr.strip()}')
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
        print_debug(f'Eval function: {func}')
        # append to execution state queue
        func = func.split('(', 1)
        name = func[0].strip()
        if len(func) == 2:
            args = func[1].strip()[:-1]
            args = split_args(args)
            for i in range(0, len(args)):
                args[i] = args[i].strip().strip('\'"')
                if args[i].startswith('_'):
                    args[i] = self.get_env_value(args[i])
        else:
            args = []

        if name == 'int':
            return MiniSpecReturnValue(int(args[0]), False)
        elif name == 'float':
            return MiniSpecReturnValue(float(args[0]), False)
        elif name == 'str':
            return MiniSpecReturnValue(args[0], False)
        else:
            skill_instance = Statement.low_level_skillset.get_skill(name)
            if skill_instance is not None:
                print_debug(f'Executing low-level skill: {skill_instance.get_name()} {args}')
                return MiniSpecReturnValue.from_tuple(skill_instance.execute(args))

            skill_instance = Statement.high_level_skillset.get_skill(name)
            if skill_instance is not None:
                print_debug(f'Executing high-level skill: {skill_instance.get_name()}', args, skill_instance.execute(args))
                interpreter = MiniSpecProgram()
                interpreter.parse([skill_instance.execute(args)])
                interpreter.finished = True
                val = interpreter.eval()
                if val.value == 'rp':
                    return MiniSpecReturnValue(f'High-level skill {skill_instance.get_name()} failed', True)
                return val
            raise Exception(f'Skill {name} is not defined')

    def eval_expr(self, var: str) -> MiniSpecReturnValue:
        print_t(f'Eval expr: {var}')
        var = var.strip()
        if var.startswith('->'):
            self.ret = True
            return MiniSpecReturnValue(self.eval_expr(var.lstrip('->')).value, True)
        if '=' in var:
            
            var, expr = var.split('=')
            print_t(f'Eval expr var assign: {var} {expr}')
            expr = expr.strip()
            ret_val = self.eval_expr(expr)
            # if not ret_val.replan:
            self.env[var] = ret_val.value
            return ret_val
        # deal with + - * / operators
        if '+' in var or '-' in var or '*' in var or '/' in var:
            if '+' in var:
                operands = var.split('+')
                val = 0
                for operand in operands:
                    val += self.eval_expr(operand).value
            elif '-' in var:
                operands = var.split('-')
                val = self.eval_expr(operands[0]).value
                for operand in operands[1:]:
                    val -= self.eval_expr(operand).value
            elif '*' in var:
                operands = var.split('*')
                val = 1
                for operand in operands:
                    val *= self.eval_expr(operand).value
            elif '/' in var:
                operands = var.split('/')
                val = self.eval_expr(operands[0]).value
                for operand in operands[1:]:
                    val /= self.eval_expr(operand).value
            return MiniSpecReturnValue(val, False)

        if len(var) == 0:
            raise Exception('Empty operand')
        if var.startswith('_'):
            return MiniSpecReturnValue(self.get_env_value(var), False)
        elif var == 'True' or var == 'False':
            return MiniSpecReturnValue(evaluate_value(var), False)
        elif var[0].isalpha():
            return self.eval_function(var)
        else:
            return MiniSpecReturnValue(evaluate_value(var), False)

    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        if '&' in condition:
            conditions = condition.split('&')
            cond = True
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan:
                    return ret_val
                cond = cond and ret_val.value
            return MiniSpecReturnValue(cond, False)
        if '|' in condition:
            conditions = condition.split('|')
            for c in conditions:
                ret_val = self.eval_condition(c)
                if ret_val.replan:
                    return ret_val
                if ret_val.value == True:
                    return MiniSpecReturnValue(True, False)
            return MiniSpecReturnValue(False, False)
        
        operand_1, comparator, operand_2 = re.split(r'(>|<|==|!=)', condition)
        operand_1 = self.eval_expr(operand_1)
        if operand_1.replan:
            return operand_1
        operand_2 = self.eval_expr(operand_2)
        if operand_2.replan:
            return operand_2
        
        print_debug(f'Condition ops: {operand_1.value} {comparator} {operand_2.value}')
        if type(operand_1.value) == int and type(operand_2.value) == float or \
            type(operand_1.value) == float and type(operand_2.value) == int:
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
        else:
            raise Exception(f'Invalid comparator: {comparator}')
        
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
        if Statement.low_level_skillset is None or \
            Statement.high_level_skillset is None:
            raise Exception('Statement: Skillset is not initialized')
        
        Statement.execution_queue = Queue()
        self.execution_thread = Thread(target=self.executor)
        self.execution_thread.start()

        self.timestamp_get_plan = None
        self.timestamp_start_execution = None
        self.timestamp_end_execution = None
        self.program_count = 0
        self.ret_queue = Queue()
        self.message_queue = message_queue

    def execute(self, code: Stream[ChatCompletion.ChatCompletionChunk] | list[str]) -> MiniSpecReturnValue:
        print_t(f'>>> Get a stream')
        self.execution_history = []
        self.timestamp_get_plan = time.time()
        program = MiniSpecProgram(mq=self.message_queue)
        program.parse(code, True)
        self.program_count = len(program.statements)
        t2 = time.time()
        print_t(">>> Program: ", program, "Time: ", t2 - self.timestamp_get_plan)

    def executor(self):
        while True:
            if not Statement.execution_queue.empty():
                if self.timestamp_start_execution is None:
                    self.timestamp_start_execution = time.time()
                    print_t(">>> Start execution")
                statement = Statement.execution_queue.get()
                print_debug(f'Queue get statement: {statement}')
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
