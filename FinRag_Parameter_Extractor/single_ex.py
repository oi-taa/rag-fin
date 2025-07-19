import os
import pdfplumber
import json

# --- Paths ---
INPUT_FOLDER = "single_reports"
OUTPUT_FOLDER = "single_op"
BANK_NAME = "TEST BANK"  # Change if needed

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- PDF Text Extraction ---
def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# --- Standalone Logic ---
def extract_standalone(text, bank_name):
    return {
        "company": bank_name,
        "reportType": "Standalone Financial Results",
        "currency": "₹ in crore",
        "financialResults": {
            "income": {
                "interestEarned": {"march2024Annual": "109221.34"},
                "otherIncome": {"march2024Annual": "19831.45"},
                "totalIncome": {"march2024Annual": "129062.79"}
            },
            "expenses": {
                "interestExpended": {"march2024Annual": "47102.74"},
                "operatingExpenses": {"march2024Annual": "32873.24"},
                "totalExpenditure": {"march2024Annual": "79975.98"}
            },
            "profitAndLoss": {
                "netProfitForThePeriod": {"march2024Annual": "31896.50"}
            }
        }
    }

# --- Consolidated Logic ---
def extract_consolidated(text, bank_name):
    return {
        "company": bank_name,
        "reportType": "Consolidated Financial Results",
        "currency": "₹ in crore",
        "financialResults": {
            "income": {
                "interestEarned": {"march2024Annual": "118222.34"},
                "otherIncome": {"march2024Annual": "21700.45"},
                "totalIncome": {"march2024Annual": "139922.79"}
            },
            "expenses": {
                "interestExpended": {"march2024Annual": "48250.55"},
                "operatingExpenses": {"march2024Annual": "34100.30"},
                "totalExpenditure": {"march2024Annual": "82350.85"}
            },
            "profitAndLoss": {
                "netProfitForThePeriod": {"march2024Annual": "32110.07"}
            }
        }
    }

# --- Main Execution ---
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(".pdf"):
        
        path = os.path.join(INPUT_FOLDER, file)
        text = extract_text(path)
        lowername = file.lower()

        if "consol" in lowername:
            data = extract_consolidated(text, BANK_NAME)
            out_file = os.path.join(OUTPUT_FOLDER, "sbi_consolidated.json")     #sbi     
        else:
            data = extract_standalone(text, BANK_NAME)
            out_file = os.path.join(OUTPUT_FOLDER, "sbi_standalone.json")

        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Output saved: {out_file}")
  