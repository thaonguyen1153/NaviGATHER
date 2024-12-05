# This is a sample Python script.
import PyPDF2
import gradio as gr
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
from termcolor import colored
import codecs
from bs4 import BeautifulSoup
import re
import spacy
import spacy_transformers
import json
from collections import defaultdict
import nltk # to process with common words, stop words, preprocess data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# model to get NER
nlp = spacy.load('en_core_web_lg')
# for text similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
#model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L12')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# get nltk
nltk.download('punkt')
nltk.download('stopwords')

def read_jd_file(filename):
    # Support functions
    """Program to read the entire file (absolute path) using read() function"""
    file = open(filename, "r")
    content = file.read()
    file.close()
    return content

def read_pdf_text(pdffile):
    """Read PDF file ans=d parse as text"""
    pdf = PyPDF2.PdfReader(pdffile)
    resume = ""

    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()
    return resume

# Comparing the similarity between the JDs
def check_jd_similarity():
    jd_1 = read_jd_file("./../CV-Job-matching-main/JD/jd_database analyst_1.txt")
    jd_2 = read_jd_file("./../CV-Job-matching-main/JD/jd_Oracle DBA_2.txt")
    jd_3 = read_jd_file("./../CV-Job-matching-main/JD/jd_Oracle DBA_3.txt")

    sentences = [jd_1, jd_2, jd_3]
    # into vector
    embeddings = model.encode(sentences)
    # compute cosine similarity
    cos_sim = util.cos_sim(embeddings, embeddings)
    # add all pairs to a list with their cosim similarity
    all_sentence_combination = []
    for i in range(len(cos_sim) - 1):
        for j in range(i + 1, len(cos_sim)):
            all_sentence_combination.append([cos_sim[i][j], i, j])

    # sort list
    all_sentence_combination = sorted(all_sentence_combination, key=lambda x: x[0], reverse=True)

    print('Sentence pair similar range: ')
    for score, i, j in all_sentence_combination:
        print(f'Comparing JD {i} with JD {j} get Score: {format(cos_sim[i][j], ".0%")}')

# Comparing the CV with each of the JD
def check_cv_vs_jd_similarity():
    jd_1 = read_jd_file("./JD/jd_database analyst_1.txt")
    jd_2 = read_jd_file("./JD/jd_Oracle DBA_2.txt")
    jd_3 = read_jd_file("./JD/jd_Oracle DBA_3.txt")

    cv1 = read_pdf_text("./CV/Akshay_Srimatrix.pdf")
    sentences = [cv1, jd_1, jd_2, jd_3]
    # into vector
    embeddings = model.encode(sentences)
    # compute cosine similarity
    cos_sim = util.cos_sim(embeddings, embeddings)
    # add all pairs to a list with their cosim similarity
    all_sentence_combination = []
    for i in range(len(cos_sim)):
        all_sentence_combination.append([cos_sim[0][i], i])

    # sort list
    all_sentence_combination = sorted(all_sentence_combination, key=lambda x: x[0], reverse=True)

    print('Sentence pair similar range: ')
    for score, i in all_sentence_combination:
        if (i > 0):
            print(f'Comparing CV with JD {i - 1} get Score: {format(cos_sim[0][i], ".0%")}')

    return all_sentence_combination, cos_sim

def calculate_similarity(file1, file2):
    sentences = [file1, file2]
    # into vector
    embeddings = model.encode(sentences)
    # compute cosine similarity
    cos_sim = util.cos_sim(embeddings, embeddings)
    # add all pairs to a list with their cosim similarity
    all_sentence_combination = []
    for i in range(len(cos_sim)):
        all_sentence_combination.append([cos_sim[0][i], i])

    # sort list
    all_sentence_combination = sorted(all_sentence_combination, key=lambda x: x[0], reverse=True)

    # print('Sentence pair similar range: ')
    #for score, i in all_sentence_combination:
    #    if (i > 0):
    #        print(f'Comparing CV with JD {i - 1} get Score: {format(cos_sim[0][i], ".0%")}')
    fscore = float(cos_sim[0][len(cos_sim)-1].detach().numpy())
    return fscore # The first one is comparing to itself

# calculate similarity for each chunk (as list)
def compare_chunks(chunk_cv, chunk_jd, is_bonus=False):
    # If both chunks are empty then return 0
    if not chunk_cv and not chunk_jd:
        return 0
    # If chunk_jd is empty and it's not a bonus comparison, return 1.0 (100% match)
    if not is_bonus and not chunk_jd:
        return 1.0


    # Convert lists to strings
    str1 = ' '.join(chunk_cv)
    str2 = ' '.join(chunk_jd)

    # Calculate similarity
    score = calculate_similarity(str1, str2)

    return score if score > 0 else 0


def calculate_similarity_chunk(str1, str2):
    # Load pre-trained model
    #model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode sentences
    embedding1 = model.encode(str1, convert_to_tensor=True)
    embedding2 = model.encode(str2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    return cosine_score.item()


def matching_score_visualization(similarity_float):
    similarity_percentage = f"{similarity_float:.1f}%"
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=similarity_float,
        mode="gauge+number",
        title={'text': "Matching percentage (%)"},
        # delta = {'reference': 100},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "#FFB6C1"},
                {'range': [50, 70], 'color': "#FFFFE0"},
                {'range': [70, 100], 'color': "#90EE90"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))

    fig.update_layout(width=600, height=400)  # Adjust the width and height as desired
    #fig.show(renderer="browser") # when returning from Gradio interface, there is no need to show here
    # Print notification
    if similarity_float < 50:
        print(colored(f"Low chance ({similarity_percentage}), need to modify your CV!", "red", attrs=["bold"]))
    elif 50 <= similarity_float < 70:
        print(colored(f"Good chance ({similarity_percentage}) but you can improve further!", "yellow", attrs=["bold"]))
    else:
        print(colored(f"Excellent! ({similarity_percentage}) You can submit your CV.", "green", attrs=["bold"]))

    return fig

def plot_2_html(fig):
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=list(map(float,a.split(' '))), y=list(map(float,b.split(' ' ))),mode='lines+markers',name='hi',marker=dict(size=10,line=dict(width=2))))

    fig.write_html("test1.html")
    f = codecs.open("test1.html",'r','utf-8')
    doc = BeautifulSoup(f, features="html.parser")
    return str(doc)

def matching_print(name):
    print('Sentence Transformer version:', sentence_transformers.__version__)

    all_sentence_combination, cos_sim = check_cv_vs_jd_similarity()
    for score, i in all_sentence_combination:
        if (i > 0):
            print(f'Comparing CV with JD {i - 1} get Score: {format(cos_sim[0][i], ".0%")}')
            #matching_score_visualization(format(cos_sim[0][i], ".0%"))
            matching_score_visualization(float(cos_sim[0][i]) * 100)

def preprocess(text):
    # Remove empty lines
    text = "".join([s for s in text.splitlines(True) if s.strip("\r\n")])
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Remove common words (you can customize this list)
    common_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
        'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
    ]
    words = [word for word in words if word not in common_words]
    # Join the words back into a string
    text = ' '.join(words)
   # don't remove numbers
    # text = re.sub('[^A-Za-z0-9\n]+', ' ', text)
    return text

def extract_ner(text, nlp):
    '''This function get the conntent of resume and label them '''
    ner_labels = []
    for doc in nlp.pipe([text], disable=["tagger", "parser"]):
      for ent in doc.ents:
          text_name = re.sub('[^A-Za-z0-9]+', ' ', ent.text).strip()
          ner_labels.append((text_name, ent.label_))
    return ner_labels


def to_json(ner_labels):
    # Create a defaultdict to group items by label
    grouped_data = defaultdict(list)

    # Group the items by their labels
    for item, label in ner_labels:
        grouped_data[label].append(item)

    # Convert the defaultdict to a regular dictionary
    json_data = dict(grouped_data)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(json_data, indent=2)
    return json_string


def get_score(cv_subset, jd_subset):
    #  join the skills in each list into a single string
    resume_text = ' '.join(cv_subset)
    jd_text = ' '.join(jd_subset)

    # create Doc objects for each
    doc_resume = nlp(resume_text)
    doc_jd = nlp(jd_text)

    # compute the similarity
    similarity_score = doc_resume.similarity(doc_jd)
    return similarity_score

def extract_by_label(ner_results, label):
    '''This function extracts skills from NER results
    label can be SKILL, EXPERIENCE, TOOL ...'''

    results = [item[0] for item in ner_results if item[1] == label]
    return results

def get_skill_score(ner_skills_cv, reg_skills_cv, ner_skills_jd):
    sscore_skill = compare_chunks(ner_skills_cv, ner_skills_jd) * 100
    sscore_skill += compare_chunks(reg_skills_cv, ner_skills_jd) * 100
    return sscore_skill

# This function extract skills from using regex, in case NER does not return the skill as expected
def extract_skills(text):
    # Find the Skills section using regex
    skills_pattern = re.compile(r'Skills:?(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)
    skills_match = skills_pattern.search(text)

    if skills_match:
        skills = skills_match.group(1).strip()
        return skills.split('\n')
    else:
        return []


def calculate_weighted_score(skill_score, experience_score, tool_score, degree_score, cross_chunk_bonus):
    # Skills: 35% (0.35) - Often the most crucial factor in job matching.
    # Experience: 30% (0.30) - Highly important, slightly less than skills.
    # Tools: 20% (0.20) - Important but generally less so than skills and experience.
    # Degree: 15% (0.15) - Relevant but usually less critical than practical skills and experience.
    weights = {
        'skill': 0.35,  # Skills are often the most important
        'experience': 0.30,  # Experience is also highly valued
        'tool': 0.20,  # Tools/technologies are important but slightly less so
        'degree': 0.15  # Degree is relevant but usually less important than skills and experience
    }

    # Ensure all scores are on the same scale (0-100)
    skill_score = min(max(skill_score, 0), 100)
    experience_score = min(max(experience_score, 0), 100)
    tool_score = min(max(tool_score, 0), 100)
    degree_score = min(max(degree_score, 0), 100)

    # Calculate weighted score
    base_score = (
            skill_score * weights['skill'] +
            experience_score * weights['experience'] +
            tool_score * weights['tool'] +
            degree_score * weights['degree']
    )

    # Add bonus score
    total_score = base_score + cross_chunk_bonus

    return min(total_score, 100)

# Define Gradio Interface
def gradio_interface(file1, file2):
    file1 = read_pdf_text(file1)
    file2 = read_pdf_text(file2)

    file1_content = preprocess(file1)
    file2_content = preprocess(file2)

    # load NER model
    # This model has to be fine tune using GPU training again with resume/jobdescription dataset
    nlp = spacy.load('./model')
    ner_labels_resume = extract_ner(file1_content, nlp)
    ner_labels_jd = extract_ner(file2_content, nlp)

    # This part is preparing the data to show on demo
    resume_json = to_json(ner_labels_resume)
    resume_parsed = json.loads(resume_json) # is a dict

    jd_json = to_json(ner_labels_jd)
    jd_parsed = json.loads(jd_json) # is a dict

    # Extract from resume
    skills_cv = extract_by_label(ner_labels_resume, 'SKILL')
    experience_cv = extract_by_label(ner_labels_resume, 'EXPERIENCE')
    tool_cv = extract_by_label(ner_labels_resume, 'TOOL')
    degree_cv = extract_by_label(ner_labels_resume, 'DEGREE')

    # Extract from jd
    skills_jd = extract_by_label(ner_labels_jd, 'SKILL')
    experience_jd = extract_by_label(ner_labels_jd, 'EXPERIENCE')
    tool_jd = extract_by_label(ner_labels_jd, 'TOOL')
    degree_jd = extract_by_label(ner_labels_jd, 'DEGREE')

    # Extract from regex:
    skill_block = extract_skills(file1)
    # Now comparing the scrore for each chunk
    sscore_skill = get_skill_score(skills_cv, skill_block, skills_jd)
    sscore_experience = compare_chunks(experience_cv, experience_jd) * 100
    sscore_tool = compare_chunks(tool_cv, tool_jd) * 100
    sscore_degree = compare_chunks(degree_cv, degree_jd) * 100

    sscore_bonus = compare_chunks(tool_cv, skills_jd, True) * 100
    print(f"BONUS Score: {sscore_bonus:.2f}")
    # Calculate the score by weights
    total_score = calculate_weighted_score(sscore_skill, sscore_experience, sscore_tool, sscore_degree, sscore_bonus)
    print(f"Total Weighted Score: {total_score:.2f}")
    plot1 = matching_score_visualization(total_score)

    # The whole similarity score of docs compare using ST
    similarity_score = calculate_similarity(file1_content, file2_content)
    plot = matching_score_visualization(similarity_score * 100)
    #plot_html = plot_2_html(plot)
    return (
        "Resume Parsing results",
        json.dumps(resume_parsed, indent=4, default=str, ensure_ascii=False),
        "Job Description Parsing results",
        json.dumps(jd_parsed, indent=4, default=str, ensure_ascii=False),
        sscore_skill,
        sscore_experience,
        sscore_degree,
        sscore_tool,
        total_score,
        plot1,
        similarity_score * 100,
        plot
    )

#if __name__ == '__main__':
#    demo.launch()
    #file1 = read_resume_text("./CV/Akshay_Srimatrix.pdf")
    #file2 = read_jd_file("./JD/jd_Oracle DBA_3.txt")
    #similarity_score = calculate_similarity(file1, file2)
    #print(f'Comparing CV with JD get Score: {format(similarity_score, ".0%")}')#[0][len(similarity_score)-1]
