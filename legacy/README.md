# Legacy scripts

This directory contains the older experiment and benchmark scripts that predate
the current RAGEval Forge flow.

The files are kept byte-for-byte as legacy code. Root-level compatibility
wrappers still forward the old commands, for example:

```powershell
python E:\RAG\new_test.py --dataset zh_int --modelname qwen3:8b
python E:\RAG\rag_retrieval_eval.py --dataset zh_int
```

Current maintained entry points remain in the project root:

- `rag_dataset_ui.py`
- `rag_dataset_builder.py`
- `mineru_ocr.py`
- `deepseekmine_compat_import.py`
- `deepseekmine_rag_eval.py`
- `run_current_retrieval_eval.py`
- `run_rag_comparison.ps1`
- `start_retrieval_eval.ps1`
