# NaviGATHER Job Matching

## Functional Requirement Description
**NAV- 1** Job matching Displays relevant and specific job opportunities
(e.g. co-op) in one consolidated job board, also
sorting the companies with their history of
diversity and inclusion

**NAV- 2** Skill-gap identification Spots the skills/ certificates that the users should
obtain for better career development in a
specific country

**NAV- 4** Personalize reject email Bring more informative and helpful information
in reject email for candidate in order to give
them insights of how their skills are not matched
with our expectations

**NAV- 3** Resume optimization Suggests improvements to tailor resumes to be
culturally adaptive and for the Applicant
Tracking Systems (ATS) and specific job
descriptions

**NAV- 5** Cover letter guidance Assists in crafting cover letters highlighting
relevant skills and experiences in the resume

## Non-Functional Requirement Description

**NAV-N 1** Security and Privacy 
• User profiles and job preferences must be
securely stored and transmitted.
- Compliance with privacy regulations (e.g., GDPR) is essential.
- Avoid storing sensitive information unnecessarily.

**NAV-N 2** Accuracy and Reliability 
• Any suggestion from the AI module is accurate (recommended jobs, recommended skill gap...)

**NAV-N 3** Scalability 
• AI module can handle a large number of data from resume dataset as well as job
dataset
   - AI module should be easy to update or expand

**NAV-N 4** Performance • Response times for job search and filtering
should meet specified thresholds.
- The resume optimization process should be efficient, especially during peak usage.
- Minimize computational resources required for AI module/method that used.

**NAV-N5** Usability • AI module provides streamlined methods
for connecting UI/UX.
- Ensure seamlessly integrates with UI/UX components without imposing its own widget or interface.

# Repository Structure

The repository is organized as follows:

- **NaviGATHER-Job-matching** main folder

- **CV folder** and **json**: This folder contains the CVs (resumes) of job candidates. It serves as the input for the matching algorithm. Trying PDF and JSON.

- **JD folder**: This folder contains the Job descriptions

- **spider_software_tagged_data**: This folder contains jobs grabed from spider

- **dataset folder**: This folder contains the job postings dataset used to train the Doc2Vec model. The dataset includes various job descriptions to create embeddings for comparison.

- **model folder**: This folder contains the model for parsing resume with label as SKILL, EXPERIENCE, TOOL....

- **cv_job_matching.model**: depricated: this model is from DOC2Vec for matching, can keep as reference to compare the results

- **requirements.txt**: This file lists all the dependencies and their respective versions required to run the matching algorithm and reproduce the results.

## Notebook

Below are description about the notebooks in this folder

1. `04_matching_usingST.ipynb` is trying to use Huggingface model for the matching
2. `05_CVParsing_Spacy.ipynb` is Parsing using NER and Spacy
3. `05_JDParsing_Spacy.ipynb` i parsing using NER and Spacy, this is included in the above file, but for next step, maybe it needs further customize
4. `06_DB_vector_resume.ipynb` updating and querying from vector database. This example use weaviate. 

Other files are trials, can be deleted later. Just keep it now for reference back.

