"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class CGECPrompter(object):
    def __init__(self, 
                 template_name: str = "llm/templates/baichuan_prompt.json"):
        with open(template_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        source: str,
        target: Union[str, None] = None,
        bos_token = None,
        eos_token = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["prompt_input"].format(source=source)
        if target:
            res = res + target
        if bos_token:
            res = bos_token + res
        if eos_token:
            res = res + eos_token
        return res
    
    def generate_align_prompt(
        self,
        source: str,
        predict: str,
        target: Union[str, None] = None,
        bos_token = None,
        eos_token = None,
    ) -> str:
        input = source + '\t' + predict
        res = self.template["align_prompt"].format(input=input)
        if target:
            res = res + target
        if bos_token:
            res = bos_token + res
        if eos_token:
            res = res + eos_token
        return res
    
    def generate_full_prompt(
        self,
        source: str,
        tag:str, 
        target: str,
        bos_token = None,
        eos_token = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["full_prompt"].format(source=source, tag=tag, target=target)
        if bos_token:
            res = bos_token + res
        if eos_token:
            res = res + eos_token
        return res
    
    def generate_input_prompt(
        self,
        source: str,
        bos_token = None,
        eos_token = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["prompt_input"].format(source=source)
        if bos_token:
            res = bos_token + res
        if eos_token:
            res = res + eos_token
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"], maxsplit=1)[1].strip()
    
    def get_tag_output(self, output: str) -> str:
        return output.split(self.template["response_tag_output"], maxsplit=1)[1].strip()
    
    def get_output(self, output: str) -> str:
        return output.split(self.template["response_output"], maxsplit=1)[1].strip()