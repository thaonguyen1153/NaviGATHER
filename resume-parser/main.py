from ResumeReader import ResumeReader
from ResumeParser import ResumeParser
from Models import Models
import json
import os


class Main:
    def __init__(self):
        models = Models()
        ner, ner_dates, zero_shot_classifier, tagger = models.load_trained_models()
        self.reader = ResumeReader()
        self.parser = ResumeParser(ner, ner_dates, zero_shot_classifier, tagger) 

    def parse_resume(self, file_path):
        resume_lines = self.reader.read_file(file_path)
        output = self.parser.parse(resume_lines)
        return output
        
    def save_parse_as_json(self, dict, file_name):
        print("Saving the parse...")
        with open(file_name, 'w', encoding="utf-8") as f:
            json.dump(dict, f, indent=4, default=str, ensure_ascii=False) 