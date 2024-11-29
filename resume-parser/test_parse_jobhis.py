import re

def extract_dates(text):
  """Extracts dates from the given text.

  Args:
    text: The input text.

  Returns:
    A list of tuples, where each tuple contains a date and its corresponding data.
  """

  dates = []
  date_pattern = r"\d{4}-\d{4}|\d{2}-\d{2}-\d{4}|\w+ \d{2}-\d{4}"
  matches = re.findall(date_pattern, text)

  for match in matches:
    start_index = text.index(match)
    end_index = start_index + len(match)
    data = text[end_index:].split("\n")[0].strip()
    dates.append((match, data))

  return dates

# Example usage:
text = """
Environment: Oracle12c, TOAD, SQLDeveloper, MSExcel, Github, Jenkins, bamboo, UNIX.
Mphasis Corporation, NY                                                                                              Dec 2016-June 2017      
Role: Oracle PL/SQL Developer 
Project: Connect Risk Engine Genesis.
...
Mastercard, O’Fallon, Missouri                                                                                        July 17-Till date      
Role: Oracle PL/SQL Developer 

 

Project: Smartdata. 
The General Data Protection Regulation (EU) ("GDPR") is a regulation in EU law on data protection and 
privacy for all individuals within the European Union (EU) and the European Economic Area 
(EEA).According to the compliance the data has to be purged in accordance with the regulation. 
"""

dates = extract_dates(text)
for date, data in dates:
  print(date, data)