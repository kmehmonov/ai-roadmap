# CLI - Command Line Interface
import sys
import argparse
from fastapi import FastAPI

if len(sys.argv) < 3:
    raise Exception()

a = int(sys.argv[1])
b = int(sys.argv[2])

print(a + b)

