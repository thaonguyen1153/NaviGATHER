# Import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk

pdf = PyPDF2.PdfReader('./CV/Akshay_Srimatrix.pdf')
resume = ""
for i in range(len(pdf.pages)):
    pageObj = pdf.pages[i]
    resume += pageObj.extract_text()

# JD by input text:
jd = """
Oracle DBA- job post
Atlantis IT group
Toronto, ON•Hybrid work
Atlantis IT group
Toronto, ON•Hybrid work
Full job description
Role: Oracle DBA

Location: Toronto, ON

Duration: 6+ months.

Must have experience in Oracle EBS R12.2.X

Expert in both technical and functional R12.2.x finance modules (preferably AP, FA, GL and Coupa)

Must have hands on experience in Oracle database and EBS R12.2.x objects.

Working experience with Oracle team on Oracle SRS for troubleshooting the technical and functional objects.

E.B Tax, functional configuration and functional testing experience is needed for this project.

Strong experience in SQL, Pl/sql, Putty, WinSCP, and Joms tools which are prerequisites.

Previous banking knowledge is a definite plus."""

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)

    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# Apply to CV and JD
input_CV = preprocess_text(resume)
input_JD = preprocess_text(jd)


# Model evaluation
model = Doc2Vec.load('cv_job_maching.model')
v1 = model.infer_vector(input_CV.split())
v2 = model.infer_vector(input_JD.split())
similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
print(round(similarity, 2))

# Visualization
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = similarity,
    mode = "gauge+number",
    title = {'text': "Matching percentage (%)"},
    #delta = {'reference': 100},
    gauge = {
        'axis': {'range': [0, 100]},
        'steps' : [
            {'range': [0, 50], 'color': "#FFB6C1"},
            {'range': [50, 70], 'color': "#FFFFE0"},
            {'range': [70, 100], 'color': "#90EE90"}
        ],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))

fig.update_layout(width=600, height=400)  # Adjust the width and height as desired
fig.show()

# Print notification
if similarity < 50:
    print(colored("Low chance, need to modify your CV!", "red", attrs=["bold"]))
elif similarity >= 50 and similarity < 70:
    print(colored("Good chance but you can improve further!", "yellow", attrs=["bold"]))
else:
    print(colored("Excellent! You can submit your CV.", "green", attrs=["bold"]))