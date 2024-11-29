from ResumeParser import ResumeParser
from ResumeReader import ResumeReader
from ResumeSegmenter import ResumeSegmenter
from Models import Models
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='resume.log', encoding='utf-8', level=logging.DEBUG)

reader = ResumeReader()
models = Models()
ner, ner_dates, zero_shot_classifier, tagger = models.load_trained_models()
parser = ResumeParser(ner, ner_dates, zero_shot_classifier, tagger)

resume_lines= reader.read_file('./Akshay_Srimatrix.pdf')
output = parser.parse(resume_lines)
print(output)