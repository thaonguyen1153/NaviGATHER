from ResumeParser import ResumeParser
from ResumeParser_woseg import ResumeParser1
from ResumeReader import ResumeReader
from ResumeSegmenter import ResumeSegmenter
from Models import Models
import PyPDF2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='resume.log', encoding='utf-8', level=logging.DEBUG)

#reader = ResumeReader()
#models = Models()
#ner, ner_dates, zero_shot_classifier, tagger = models.load_trained_models()
#parser = ResumeParser(ner, ner_dates, zero_shot_classifier, tagger)

#resume_lines= reader.read_file('./Akshay_Srimatrix.pdf')
#output = parser.parse(resume_lines)
#print(output)


def build_resume_vector(pdffile):
    """Read PDF file ans=d parse as text"""
    pdf = PyPDF2.PdfReader(pdffile)
    resume = ""

    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()
    return resume

resume_lines = build_resume_vector('./Akshay_Srimatrix.pdf')

parser = ResumeReader1(resume_lines)
parser.get_extracted_data()