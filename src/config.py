import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
MAX_RETRIES = 10       # max revise-answer retries in the IsSUP loop
MAX_REWRITE_TRIES = 3  # max rewrite-question retries in the IsUSE loop