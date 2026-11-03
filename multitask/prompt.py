"""One instruction wrapper shared by multitask training and later evaluation."""
import json

PROMPT_VERSION='quoted_instruction_v1'

def instruction_prompt(instruction):
    instruction=instruction.strip().lower()
    assert instruction and '\n' not in instruction
    return 'What action should the robot take to follow this instruction: '+json.dumps(instruction,ensure_ascii=False)+'?'
