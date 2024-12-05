import gradio as gr
from main import gradio_interface
import warnings
warnings.filterwarnings("ignore")

def demo_function(file_resume, file_jd):
    return gradio_interface(file_resume, file_jd)

description = "Demo for parsing resume for **NaviGATHER**. "
article = "<h3>How to Use:</h3> " \
          "<ul><li>Upload the resume in the top panel</li> " \
          "<li>Upload the job description in the below panel</li> " \
          "<li>Click on the 'Submit' button. </li>" \
          "<li><strong>Great!</strong>. Check if all required sections are chunk and recognize in json panels on the right, the score will be shown at the bottom</li></ul>"

demo = gr.Interface(
        fn=demo_function,
        inputs=[
            gr.File(label="Upload CV as PDF", file_count="single", file_types=[".pdf"]),
            gr.File(label="Upload Job Description as PDF", file_count="single", file_types=[".pdf"])
            #gr.Textbox(label="Enter Job Description", lines=7, placeholder="Paste or type the job description here...")
        ],
        outputs=[
            gr.Markdown(label="Resume Parsing"),
            gr.JSON(),
            gr.Markdown(label="Job Description Parsing"),
            gr.JSON(open=True, show_indices=True),
            gr.Number(label="Skill Score: "),
            gr.Number(label="Experience Score: "),
            gr.Number(label="Degree Score: "),
            gr.Number(label="Tool Score: "),
            gr.Number(label="Total Score"),
            gr.Plot(label="Total Score Visualization", format="png"),
            gr.Number(label="Similarity Score"),
            gr.Plot(label="Similarity Visualization", format="png")
        ],
        title="Resume - Job Description Similarity",
        theme=gr.themes.Glass(),
        description=description,
        article=article,
        flagging_mode="never"  # This removes the flag button
    )
demo.launch()