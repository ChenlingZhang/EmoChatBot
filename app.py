import streamlit as st
import torch
import logging
import models

st.title("Emotion ChatBot")
DEFAULT_SYS_PROMPUT = "You are a helpful assistant, you should answer the question correctly."

model_desc = ("DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
              "DeBERTa improves the BERT and RoBERTa models using disentangled attention"
              "and enhanced mask decoder. It outperforms BERT and RoBERTa on majority of NLU tasks with 80GB training "
              "data.")

# with st.sidebar:
#     st.markdown("### Choose your model")
#     select_models = st.selectbox(
#         "Which model would you like to be use",
#         ("sileod/deberta-v3-base-tasksource-nli",
#          "bart-large-mnli",
#          "microsoft/deberta-base-mnli")
#     )
#     if select_models == "bart-large-mnli":
#         model_desc = (
#             "Bart-large-mnli is a transformer model developed by Facebook that can perform natural language generation,"
#             "translation, and comprehension tasks. It is based on the BART architecture, which combines a bidirectional"
#             "encoder and a causal decoder, and it is fine-tuned on the MultiNLI (MNLI) dataset, which is a collection "
#             "of"
#             "sentence pairs annotated with natural language inference labels1. You can use this model for zero-shot "
#             "text classification,"
#             "which is the task of assigning labels to text without any training data2.")
#
#     st.markdown("### Model Descriptions")
#     st.markdown(model_desc)

# 此处需要将用户选择的model传入 模型选择器
# current_model = select_models
# logging.info(f"current_classify_model {current_model}")

# store llm generate response
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How may I help you"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# user provide prompt
if not torch.cuda.is_available():
    st.error("Running on CPU 🥶 This demo does not work on CPU.")

if user_prompt := st.chat_input(disabled=not torch.cuda.is_available()):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

# Generate Model Response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = models.generate_response(user_prompt,chat_history=st.session_state.messages)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

