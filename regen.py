#!/usr/bin/env python3

# based off the code at https://github.com/minimaxir/aitextgen#quick-examples

__author__ = "aidswidjaja"
__version__ = "0.1.0"
__license__ = "MIT"

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

config = GPT2ConfigCPU()


#ai2 = aitextgen(model_folder="trained_model",
#                tokenizer_file="aitextgen.tokenizer.json")

ai2 = aitextgen()

ai2.generate_to_file(n=1, min_length=750, prompt="The same, but only a little different.")
