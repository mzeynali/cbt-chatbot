from itertools import count
from turtle import onclick
from streamlit_chat import message
import streamlit as st
import openai
from tkinter.messagebox import QUESTION
import json
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain import PromptTemplate
import os
from numpy.random import choice
from streamlit_autorefresh import st_autorefresh
from numpy.random import randint

# def generate_response(prompt):
#     completions = openai.Completion.create (
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=1024,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )

#     return completions

intensity = None

def generate_response(question_input: str):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"{question_input}"
            },
        ])


st.title("🤖 CBT Chatbot 🤖")

os.environ["OPENAI_API_KEY"] = "Your_API-KEY"

prompt_qa = """
Please compare the following question and answer and indicate if they are related:
Question: {text1}
Answer: {text2}
Answer with 'yes' if the texts are related or 'no' if they are not.
"""

model_name = "gpt-3.5-turbo"
# model_name = "text-davinci-003"
llm = OpenAI(temperature=0.0, model_name=model_name)

prompt_template_qa = PromptTemplate(
    input_variables=["text1", "text2"], template=prompt_qa
)

with open('questions.json', 'r') as fp:
    questions = json.load(fp)['questions']


if 'Therapist' not in st.session_state:
    st.session_state['Therapist'] = []
    st.session_state['Teenager'] = []
    dialogflow = ''

def proc():
    st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

import contextlib
idx = len(st.session_state.Therapist)
# print(f"idx: {idx}")

question = choice(questions[idx])
therapist_input = st.text_input(
    "Therapist: :male-doctor:", question, key="therapist_input", disabled=True)
teenager_input = st.text_input("Teenager:", "", key="teenager_input")


print(f"Therapist: {st.session_state.Therapist}")
print(f"Teenager: {st.session_state.Teenager}")

with contextlib.suppress(Exception):
    st.button("Next Question", key="send", on_click=proc)
    is_related = llm(prompt_template_qa.format(text1=question, text2=teenager_input))

    if "Yes" in is_related and len(teenager_input) != 0:
        st.session_state.Teenager.append(teenager_input)
        st.session_state.Therapist.append(therapist_input)

        if st.session_state['Therapist']:
            if len(st.session_state['Teenager'][0]) == 0:
                st.session_state['Teenager'].pop(0)

            for i in range(len(st.session_state['Teenager'])):
                message(st.session_state["Therapist"][i], key=str(i))
                message(st.session_state['Teenager'][i],
                        is_user=True, key=f'{str(i)}_user')

        if idx == len(questions) - 1:
            dialogflow = ''
            for i in range(len(st.session_state['Teenager'])):
                dialogflow += f"Therapist: {st.session_state['Therapist'][i]}" + "\n"
                dialogflow += f"Teenager: {st.session_state['Teenager'][i]}" + "\n\n"

            prompt_intensity = f"""
        The below conversation is related to the dialog flow between the therapist and the teenager. 
        classify the below conversation into the intensity of cognitive distortion. JUST ONLY answer in short. DON'T give a summary of the conversation. 
        conversation: {dialogflow}
        JUST ONLY answer in short. DON'T give a summary of the conversation.
        """
            intensity = generate_response(prompt_intensity)
            st.success(intensity.choices[0].message.content.lstrip())

        if intensity:

            prompt_tokens = intensity["usage"]["prompt_tokens"]
            completion_tokes = intensity["usage"]["completion_tokens"]
            total_tokens_used = intensity["usage"]["total_tokens"]

            cost_of_response = total_tokens_used * 0.000002

            with st.sidebar:
                st.title("Usage Stats:")
                st.markdown("""---""")
                st.write("Promt tokens used :", prompt_tokens)
                st.write("Completion tokens used :", completion_tokes)
                st.write("Total tokens used :", total_tokens_used)
                st.write("Total cost of request: ${:.8f}".format(cost_of_response))
    else:
        if len(teenager_input) != 0:
            st.error('Not Related')
        if st.session_state['Therapist']:
            if len(st.session_state['Teenager'][0]) == 0:
                st.session_state['Teenager'].pop(0)

            for i in range(len(st.session_state['Teenager'])):
                message(st.session_state["Therapist"][i], key=str(i))
                message(st.session_state['Teenager'][i],
                        is_user=True, key=f'{str(i)}_user')
