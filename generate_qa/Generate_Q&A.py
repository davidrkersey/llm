######################################################
##### Generate Q & A from Docs for Fine Tuning #######
######################################################

"""
The following script can be used to create questions and answers for a set of documents (PDF, DOCX, CSV) to generate training data for fine tunning a LLM 
for your Q&A application. It leverages the OpenAI API to generate questions and answers from snippets of your document collection. The following are parameters
to keep in mind:

- chunk_size : dictates how long each snippet will be. Larger snippets will provide more context but will be more expensive (more tokens).
- sample_size: dictates how many Q&As will be generated. The script randomly pulls n-samples from your processed corpus. Larger n_samples means more questions 
             but will incur more cost.

"""

import os
from PyPDF2 import PdfReader
import re
import docxpy
import random
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

"""
Specify API Keys
"""

"""
Specify preprocessing
"""
convert_txt = False
chunk_txt = True

"""
Specify filepaths
"""

def get_filepath():
  """Returns the filepath of the directory"""
  filepath = os.path.dirname(os.path.realpath(__file__))
  return filepath

# Get filepaths
main_dir = get_filepath()
doc_dir = os.path.join(main_dir, 'docs')
clean_dir = os.path.join(main_dir, 'docs_clean')
out_dir = os.path.join(main_dir, 'output')

# Reset directory
#if os.path.exists(clean_dir):
#    os.rmdir(clean_dir)
#    os.makedirs(clean_dir)
#else:
#   os.makedirs(clean_dir)

if os.path.exists(out_dir):
    os.rmdir(out_dir)
    os.makedirs(out_dir)
else:
   os.makedirs(out_dir)

"""
Text Processing

The files in PDF, docx and CSV in the 'docs' directory will be converted to .txt format and stored in the 'docs_clean' folder

Add toggles for 'cleaning the text'
"""

def convert_csv_to_txt(csv_file, txt_file):
    """Converts a CSV file to a TXT file.

    Args:
        csv_file: The path to the CSV file.
        txt_file: The path to the TXT file.

    Returns:
        None.
    """

    with open(csv_file, "r") as csv_file:
        with open(txt_file, "w") as txt_file:
            for line in csv_file:
                txt_file.write(line)

if convert_txt:
    
    text_ls = []

    """
    Iterate through all pdfs and docx files and convert them to text. 
    Save them to a list as a tuple with the document file location.
    """
    for i in os.listdir(doc_dir):
        filename = os.path.join(doc_dir, i)
        if re.search('.pdf', filename) is not None:
            print(filename)
            text = ""
            with open(filename, 'rb') as f:
                reader = PdfReader(f)
                for pg in reader.pages:
                    text += pg.extract_text()
                text = text.strip()
                #text = text.replace("\n", "")
                #text = text.replace("\t", "")
                #text = text.replace("  ", " ")
                text_ls.append((filename, text))
        elif re.search('.docx', filename) is not None:
            if re.search('.docx', filename) is not None:
                print(filename)
                text = docxpy.process(filename)
                text = text.strip()
                #text = text.replace("\n", "")
                #text = text.replace("\t", "")
                #text = text.replace("  ", " ")
                text_ls.append((filename, text))
        elif re.search('.csv', filename) is not None:
            if re.search('GFEBS_FAQ', filename) is not None:
                with open(filename, "r", encoding="cp1252") as csv_file:
                    print(filename)
                    text= csv_file.read()
                    #print(text)
                    text_ls.append((filename, text))
            else:
                with open(filename, "r", encoding="utf-8") as csv_file:
                    print(filename)
                    text= csv_file.read()
                    #print(text)
                    text_ls.append((filename, text))
        else: 
            pass


    """
    Replace the file path to the cleaned docs directory and replace .pdf or .docx with .txt.
    Save the file to that location.
    """
    for i in text_ls:
        if re.search('.pdf', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".pdf", ".txt")
            print(filepath)
        if re.search('.docx', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".docx", ".txt")
            print(filepath)
        if re.search('.csv', i[0]) is not None:
            filepath = i[0].replace(doc_dir,clean_dir)
            filepath = filepath.replace(".csv", ".txt")
            print(filepath)
        with open(filepath, 'w+', encoding="utf-8") as floc:
            floc.write(i[1])


def split_into_chunks(text, chunk_size):
  """
  Splits text into chunks
  """
  chunks = []
  for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])
  return chunks


if chunk_txt:
    ## Specify chunk size. Larger chunks means more context. Smaller is less.
    chunk_size = 1000
    chunk_ls = []

    ## Loop to iterate over all documents in directory and break them into n-size chunks
    for file in os.listdir(doc_dir):
        filename = os.path.join(doc_dir, file)
        text = open(filename, 'r', errors='ignore').read()
        chunks = split_into_chunks(text, chunk_size)
    for i in chunks:
        chunk_ls.append(i)


"""
Specify number of questions by taking random sample of chunks
"""

sample_size = 3

ls_rand = random.sample(chunk_ls, sample_size)

"""
Function to generate question and answers
"""

def call_model(prompt_template, input_vars(list), chunk_ls, gen_q = True):
    
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=prompt_template,
        )
    
    model = OpenAI(openai_api_key = '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')
    chain = LLMChain(prompt=prompt, llm=model)

    if gen_q:
        df = pd.DataFrame(chunk_ls, columns=['content'])
        df['question'] = ''

        for i in range(len(df)):
            df.loc[i, 'question'] = chain.run(df.loc[i, "content"])
    
    else:
        df['answer'] = ''

        for i in range(len(df)):
            df.loc[i, 'answer'] = chain.run({'content':df.loc[i, "content"], 'question':df.loc[i, "question"]})

    return df

"""
Generate questions from chunk of text
"""

## Prompt for generating a question from a chunk of text
qa_gen_template = """
You will be generating questions based on content. Use the following content (delimited by <ctx></ctx>) and only the following content to formulate a question:
-----
<ctx>
{content}
</ctx>
-----
Answer:
)
"""

## Prompt node
prompt = PromptTemplate(
    input_variables=["content"],
    template=qa_gen_template,
)

## Model specification
model = OpenAI(openai_api_key = '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')
chain = LLMChain(prompt=prompt, llm=model)

# Define HF Model
#HUGGING_FACE_API_KEY = "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"

#Model
#model = HuggingFaceHub(repo_id="gpt2",
#                       model_kwargs={"temperature": 0, "max_length":100},
#                       huggingfacehub_api_token=HUGGING_FACE_API_KEY)
#chain = LLMChain(prompt=prompt, llm=model)

#Generate dataframe with content, question
df = pd.DataFrame(ls_rand, columns=['content'])
df['question'] = ''

#Run model
for i in range(len(df)):
  df.loc[i, 'question'] = chain.run(df.loc[i, "content"])


"""
Generate answer from chunk and question
"""

## Prompt for generating a answer from a question and chunk of text
qa_answer_template = """
You will be answering questions based on content. Use the following content (delimited by <ctx></ctx>) and the question (delimited by <que></que>) to formulate an answer:
-----
<ctx>
{content}
</ctx>
-----
<que>
{question}
</que>
-----
Answer:
)
"""

#Prompt node
prompt = PromptTemplate(
    input_variables=["content","question"],
    template=qa_answer_template,
)

## Model specification
model = OpenAI(openai_api_key = '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')
chain = LLMChain(prompt=prompt, llm=model)

#Dataframe
df['answer'] = ''

#Run model
for i in range(len(df)):
  df.loc[i, 'answer'] = chain.run({'content':df.loc[i, "content"], 'question':df.loc[i, "question"]})

"""
Save Model Outputs
"""

df.to_csv(out_dir)