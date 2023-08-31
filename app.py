import streamlit as st
import openai
from transformers import pipeline

## web api key = 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
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant, you should answer the question correctly."
emotion_classify = pipeline("sentiment-analysis")

with st.sidebar:
    openai_api_key = st.text_input("Open AI Key", key="chat_key", type="password")
    "[Generate An OpenAI API key](https://platform.openai.com/account/api-keys)"

st.title("Emotion Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please enter a valid api key to continue")
        st.stop()

    emo_label = emotion_classify(user_prompt)[0]["label"]

    if emo_label == 'NEGATIVE':
        print("negative emotion detacted")
        adjusted_system_prompt = (
            f'You are a helpful assistant, you should answer the question correctly,'
            f' also provide some advice for {emo_label} emotion start with "I think you are feeling {emo_label},'
            f'here are some suggestions for you"'
        )
    else:
        adjusted_system_prompt = DEFAULT_SYSTEM_PROMPT

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    # Construct history messages including the adjusted system prompt
    history = [
        {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history
    )

    response_msg = response.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": response_msg["content"]})
    st.chat_message("assistant").write(response_msg["content"])