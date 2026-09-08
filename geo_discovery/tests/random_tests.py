import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion import ingest_doc, ingest_url, ingest_folder, RawSource
import subprocess, sys, os


src = ingest_url("https://www.nestle.in/")
print(repr(src.text))        # ← you'll see the 175 chars = cookie/JS notice
print(src.meta)              # status, title