#  Financial Parameter Extractor for Indian Bank Reports

This repository provides a modular pipeline to extract risk-relevant paragraphs from quarterly financial disclosures issued by Indian banks. It supports both multi-bank batch processing and focused single-bank extraction.

---
```markdown
## 📁 Folder Structure
1.├── multi_bankextractor.py # Batch processor for all reports 
  ├── single_Ex.py # Targeted extractor for one PDF 
  ├── requirements.txt # Dependency list
  
2.├── bank_reports/ # Input PDFs 
  │ ├── consolidated/ # Consolidated financial reports 
  │ └── standalone/ # Standalone bank results

3.├── outputs/ # Extraction outputs 
  │ ├── consolidated/ # JSON outputs for consolidated PDFs 
  │ └── standalone/ # JSON outputs for standalone PDFs

4.├── single_ex.py # single processor for testing (single_debug.py or single_test.py).

5.├── single_op/ # Output from single-bank extractor 
  │ └── SBI_Q4_2024.json # Sample output
```


## How It Works

This extractor system processes Indian bank financial reports—both in batch and targeted modes—to produce structured JSON outputs of risk-relevant parameters\paragraphs.

###  Multi-Bank Extraction (`multi_bankextractor.py`)

The batch extractor scans all PDFs inside `bank_reports/` (both `consolidated/` and `standalone/`) and executes a rule-based paragraph segmentation pipeline.

- Identifies key performance indicators like NPA, RoA, EPS, CRAR.
- Differentiates consolidated vs. standalone reports using internal markers.
- Maps extracted paragraphs to a standardized JSON schema.
- Saves outputs in the `outputs/` folder, organized by format.
- Any PDF with structural parsing issues is moved to `bugbin/` for manual review.

###  Single-Bank Testing (`single_Ex.py`)

If you need to inspect or debug a particular bank’s report manually, run the single extractor script:

- Takes one PDF as input (full file path required)
- Extracts paragraph-level performance data using the same schema
- Output is saved inside `single_op/` for easy verification
- Works well for development, analysis spot-checking, or edge-case tuning
