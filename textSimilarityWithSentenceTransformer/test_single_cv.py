import PyPDF2
import re

def read_pdf_text(pdffile):
    """Read PDF file ans=d parse as text"""
    pdf = PyPDF2.PdfReader(pdffile)
    resume = ""

    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()
    return resume

def extract_skills(pdf_path):
    pdf = PyPDF2.PdfReader(pdf_path)
    text = ""
    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        text += pageObj.extract_text()

    # Find the Skills section using regex
    skills_pattern = re.compile(r'Skills:?(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)
    skills_match = skills_pattern.search(text)

    if skills_match:
        skills = skills_match.group(1).strip()
        return skills.split('\n')
    else:
        return []


# Usage
pdf_path = "./CV/Amelia Sanches_Updated.pdf"
#file1 = read_pdf_text("./CV/Amelia Sanches_Updated.pdf")
skills = extract_skills(pdf_path)
print(skills)
