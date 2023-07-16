import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

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


def split_into_chunks(text, chunk_size):
  """
  Splits text into chunks

  Args:
        text: String text to be split into chunks.
        chunk_size: The size of each chunk.

    Returns:
        chunks: List of text chunks.
  """
  chunks = []
  for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])
  return chunks


def call_model(df,prompt_template, input_vars, chunk_ls, openai_api, gen_q = True):
    
    """Generates questions and answers from a randomly sampled list of text from documents.

    Args:
        df: Pandas dataframe with text chunks.
        prompt_template: The prompt fed to OpenAI, either question or answer generating prompt.
        input_vars: List of variables that are fed as arguments to the prompt.
        chunk_ls: List of text chunks.
        openai_api: OpenAI API key.
        gen_q: Boolean value to determine whether to generate questions or answers.

    Returns:
        df: Pandas dataframe with questions and answers.
    """
    
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=prompt_template,
        )
    
    model = OpenAI(openai_api_key = openai_api)
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
