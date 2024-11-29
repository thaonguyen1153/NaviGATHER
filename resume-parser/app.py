from pydoc import describe
import gradio as gr
from main import Main


main = Main()

def parse_resume(resume):
    return main.parse_resume(resume.name)
    

description = "Demo for parsing resume for **NaviGATHER**. "
article = "Base on HuggingFace resume-parser" \
        "<h3>How to Use:</h3> " \
          "<ul><li>Upload the resume on the left panel</li> " \
          "<li>Click on the 'Submit' button. </li>" \
          "<li><strong>Great!</strong>. Check if all required sections are chunk and recognize... </li></ul>"
file_input = gr.File(file_count="single", label="Upload a Resume: .PDF Or .TXT")
iface = gr.Interface(fn=parse_resume, inputs=file_input, outputs="json", flagging_mode="never",
                    title="Resume Parser", theme=gr.themes.Glass(), description=description, article=article)

iface.launch()