# DocVQA Setup

This repo now includes a TraceRAG-side preparation step for DocVQA.

## What you download

You still download the DocVQA dataset yourself. TraceRAG expects:

- a questions file (`.json` or `.jsonl`)
- a directory of document images
- optionally an OCR directory

## What TraceRAG prepares

`tracerag-docvqa-prepare` will:

1. resolve DocVQA questions to image files
2. ingest OCR into TraceRAG `VectorObject`s
3. build structural artifacts under `index_root/structural/...`
4. build visual patch grids under `index_root/visual/...`
5. build `text_index.pkl`
6. write a TraceRAG-ready benchmark `.jsonl`

## Example

```powershell
tracerag-docvqa-prepare `
  --questions-path C:\data\docvqa\val\questions.json `
  --images-dir C:\data\docvqa\documents `
  --ocr-dir C:\data\docvqa\ocr `
  --dataset-root C:\data\docvqa `
  --output-index-root C:\data\tracerag_docvqa_index `
  --output-benchmark-path C:\data\tracerag_docvqa\docvqa_val.jsonl `
  --split val
```

Then run evaluation:

```powershell
tracerag-eval `
  --benchmark docvqa `
  --index-root C:\data\tracerag_docvqa_index `
  --data-path C:\data\tracerag_docvqa\docvqa_val.jsonl `
  --output-file C:\data\tracerag_docvqa\docvqa_val_results.json
```

## OCR formats supported

The prep command accepts several common OCR layouts:

- Azure-style `recognitionResults[].lines[].words[]`
- `analyzeResult.readResults/pages`
- flat `words` arrays with `text` + `bbox/box/polygon`
- plain `.txt` OCR dumps

If OCR is missing, TraceRAG falls back to a single image object, but retrieval quality will be much worse.

## Notes

- Each DocVQA image is treated as a single-page document with page id `doc_id_v1_p0`.
- The evaluator reports both exact match and ANLS.
- The benchmark loader is flexible enough to read either the prepared TraceRAG JSONL or many raw DocVQA-style question exports.
