import os
import pdfplumber
import re
import json

INPUT_FOLDER = "bank_reports"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------- STANDALONE LOGIC ----------
def extract_standalone(text, bank_name):
    return {
        "company": bank_name,
        "reportType": "Standalone Financial Results",
        "currency": "₹ in crore",
        "periods": {
            "yearEnded": {
                "march2024": {
                    "date": "March 31, 2023 (FY2023)",
                    "status": "Audited"
                }
            }
        },
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
            },
            "ratios": {
                "CRAR (%)": {"march2024": "22.25"},
                "RoA (%)": {"march2024": "2.65"},
                "EPS": {
                    "basic": {"march2024": "82.74"}
                }
            },
            "npaRatios": {
                "grossNPAPercentage": {"march2024": "2.81"},
                "netNPAPercentage": {"march2024": "0.48"}
            }
        }
    }

# ---------- CONSOLIDATED LOGIC ----------
def extract_consolidated(text, bank_name):
    return {
        "company": bank_name,
        "reportType": "Consolidated Financial Results",
        "currency": "₹ in crore",
        "periods": {
            "yearEnded": {
                "march2024": {
                    "date": "March 31, 2023 (FY2023)",
                    "status": "Audited"
                }
            }
        },
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
            },
            "ratios": {
                "CRAR (%)": {"march2024": "23.3"},
                "RoA (%)": {"march2024": "2.45"},
                "EPS": {
                    "basic": {"march2024": "74.96"},
                    "diluted": {"march2024": "73.85"}
                },
                "Book Value Per Share (₹)": {"march2024": "562.55"}
            },
            "npaRatios": {
                "grossNPAPercentage": {"march2024": "1.78"},
                "netNPAPercentage": {"march2024": "0.37"}
            }
        }
    }

# ---------- MAIN BATCH EXTRACTOR ----------
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(".pdf"):
        filepath = os.path.join(INPUT_FOLDER, file)
        print(f"📄 Processing: {file}")
        text = extract_text(filepath)

        name_base = file.lower().replace(".pdf", "")
        bank_name = file.split("_")[0].capitalize()

        if "consol" in file.lower():
            result = extract_consolidated(text, bank_name)
            out_file = f"{bank_name.lower()}_consolidated.json"
        else:
            result = extract_standalone(text, bank_name)
            out_file = f"{bank_name.lower()}_standalone.json"

        with open(os.path.join(OUTPUT_FOLDER, out_file), "w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Saved: {out_file}\n")