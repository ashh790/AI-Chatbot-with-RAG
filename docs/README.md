# Knowledge base source documents

Drop the files you want the chatbot to answer from in this folder, then run:

```bash
python ingest.py
```

Supported: `.pdf` `.txt` `.md` `.markdown` `.csv` `.json` `.log`

Subfolders are scanned too. Re-running is safe — chunks are upserted by a
stable id, so an unchanged file just overwrites itself rather than duplicating.

Useful flags:

```bash
python ingest.py --stats            # how many chunks are indexed
python ingest.py --reset            # wipe and rebuild from scratch
python ingest.py --path notes.pdf   # ingest one specific file
```

Note: scanned PDFs (images of text) extract nothing. Run OCR on them first.
