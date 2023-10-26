from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login
import torch
import logging

logging.basicConfig(filename='./logs/running_log.log', filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.INFO)

login(token="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")

debart_model_name = "sileod/deberta-v3-base-tasksource-nli"
bart_model_name = "facebook/bart-large-mnli"
mdebart_model_name = "microsoft/deberta-base-mnli"

debart_tokenizer = AutoTokenizer.from_pretrained(debart_model_name)
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
mdebart_tokenizer = AutoTokenizer.from_pretrained(mdebart_model_name)

bart_model = AutoModelForSequenceClassification.from_pretrained(bart_model_name)
debart_model = AutoModelForSequenceClassification.from_pretrained(debart_model_name)
mdebart_model = AutoModelForSequenceClassification.from_pretrained(mdebart_model_name)

bart_model_classifier = pipeline("zero-shot-classification", model=bart_model, tokenizer=bart_tokenizer)
debart_model_classifier = pipeline("zero-shot-classification", model=debart_model, tokenizer=debart_tokenizer)
mdebart_model_classifier = pipeline("zero-shot-classification", model=debart_model, tokenizer=debart_tokenizer)

DEFAULT_SYS_PROMPUT = "You are a helpful assistant, you should answer the question correctly."

llama_model_id = "meta-llama/Llama-2-7b-chat-hf"
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        torch_dtype=torch.float16,
        device_map='auto'
    )
else:
    model = None
llama_model_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)


def emotion_classify(user_prompt):
    # logging.info(f"Start create emotion label with mode {model_name}")
    emotion_labels = ["natural", "happy", "sad", "anger", "hate"]
    predict_list = []
    count_sad = 0
    count_happy = 0

    predict1 = mdebart_model_classifier(user_prompt, candidate_labels=emotion_labels)
    predict2 = bart_model_classifier(user_prompt, candidate_labels=emotion_labels)
    predict3 = debart_model_classifier(user_prompt, candidate_labels=emotion_labels)

    logging.info(f"predict1: {predict1['labels'][0]}")
    logging.info(f"predict2: {predict2['labels'][0]}")
    logging.info(f"predict3: {predict3['labels'][0]}")

    predict_list.append(predict1['labels'][0])
    predict_list.append(predict2['labels'][0])
    predict_list.append(predict3['labels'][0])

    for label in predict_list:
        if label != 'happy':
            count_sad += 1
        else:
            count_happy += 1

    if count_sad >= count_happy:
        return 'negative emotion'
    else:
        return 'positive emotion'


    # if model_name == "microsoft/deberta-base-mnli":
    #     text = user_prompt
    #     prediction = mdebart_model_classifier(text, candidate_labels=emotion_labels)
    #     emotion_label = prediction['labels'][0]
    #     logging.info(f"/f current classify model {model_name}, emotion label {emotion_label}")
    #     return emotion_label
    #
    # elif model_name == "bart-large-mnli":
    #     text = user_prompt
    #     prediction = bart_model_classifier(text, candidate_labels=emotion_labels)
    #     emotion_label = prediction['labels'][0]
    #     logging.info(f"current classify model {model_name}, emotion label {emotion_label}")
    #     return emotion_label
    #
    # elif model_name == "sileod/deberta-v3-base-tasksource-nli":
    #     text = user_prompt
    #     prediction = debart_model_classifier(text, candidate_labels=emotion_labels)
    #     emotion_label = prediction['labels'][0]
    #     logging.info(f"/f current classify model {model_name}, emotion label {emotion_label}")
    #     return emotion_label
    #
    # else:
    #     logging.info(f"no matches model")


def get_sys_prompt(user_prompt) -> str:
    emotion_label = emotion_classify(user_prompt)
    logging.info(f'current emotion label {emotion_label}')
    # good_emotion_list = ['nature', 'happy']
    if emotion_label != "positive emotion":
        DEFAULT_SYS_PROMPUT = (
           f" You take on the role of a mental health assistant. Your work environment is within the campus. \
           Please note that this conversation should be in a multi-turn question-and-answer format\
           Assist users in gaining a deeper understanding of and coming to terms with {emotion_label}\
           Analyze the factors contributing to {emotion_label}\
           demonstrating empathy for the user's emotional state\
           You can draw inspiration from cognitive behavioral therapy\
           combine it with your own expertise as a professional psychologist"
        )
        logging.info(f"current prompt {DEFAULT_SYS_PROMPUT}")
        return DEFAULT_SYS_PROMPUT
    else:
        DEFAULT_SYS_PROMPUT = (
            f"You take on the role of a mental health assistant. Your work environment is within the campus. \
           Please note that this conversation should be in a multi-turn question-and-answer format\
           Pay attention to understanding people's {emotion_label} in guided\
           conversations to help define problemsgi and generate solutions.")

    return DEFAULT_SYS_PROMPUT


def get_prompt(user_prompt: str, chat_history) -> str:
    system_prompt = get_sys_prompt(user_prompt)
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, responses in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {responses.strip()} </s><s>[INST] ')
    user_prompt = user_prompt.strip() if do_strip else user_prompt
    texts.append(f'{user_prompt} [/INST]')
    result = ''.join(texts)
    logging.info(f"current full prompt:{result}")
    return result


def generate_response(user_prompt, chat_history):
    prompt = get_prompt(user_prompt, chat_history)
    inputs = llama_model_tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    streamer = TextIteratorStreamer(llama_model_tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=1.25,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    outputs = []
    for text in streamer:
        outputs.append(text)
    result = ''.join(str(e) for e in outputs)
    return result
