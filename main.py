#!/usr/bin/env python3

# based off the code at https://github.com/minimaxir/aitextgen#quick-examples

__author__ = "aidswidjaja"
__version__ = "0.1.0"
__license__ = "MIT"

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

# The name of the sample size text for training
file_name = "input.txt"

# Train a custom BPE Tokenizer on the downloaded text
# This will save one file: `aitextgen.tokenizer.json`, which contains the
# information needed to rebuild the tokenizer.
train_tokenizer(file_name)
tokenizer_file = "aitextgen.tokenizer.json"

# GPT2ConfigCPU is a mini variant of GPT-2 optimized for CPU-training
# e.g. the # of input tokens here is 64 vs. 1024 for base GPT-2.

# that poor Nvidia G30M
#config = GPT2ConfigCPU()

# Instantiate aitextgen using the created tokenizer and config
ai = aitextgen(tokenizer_file=tokenizer_file, config=config)

# You can build datasets for training by creating TokenDatasets,
# which automatically processes the dataset with the appropriate size.
data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)

# Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
ai.train(data, batch_size=8, num_steps=50000, generate_every=5000, save_every=5000)

# Generate text from it!
ai.generate(10, prompt="The same, but only a little different.")

# With your trained model, you can reload the model at any time by
# providing the folder containing the pytorch_model.bin model weights + the config, and providing the tokenizer.
ai2 = aitextgen(model_folder="trained_model",
                tokenizer_file="aitextgen.tokenizer.json")

ai2.generate(10, prompt="The same, but only a little different.")

ai3 = aitextgen(model_folder="trained_model",
                tokenizer_file="aitextgen.tokenizer.json")
                
ai2.generate(10, prompt="The same, but only a little different.")