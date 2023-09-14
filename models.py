from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login
import torch
import logging

logging.basicConfig(filename='./logs/running_log.log', filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)

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


def emotion_classify(model_name, user_prompt):
    emotion_labels = ["natural", "happy", "sad", "anger", "hate"]
    if model_name == "microsoft/deberta-base-mnli":
        text = user_prompt
        prediction = mdebart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        logging.info(f"/f current classify model {model_name}, emotion label {emotion_label}")
        return emotion_label

    if model_name == "bart-large-mnli":
        text = user_prompt
        prediction = bart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        logging.info(f"current classify model {model_name}, emotion label {emotion_label}")
        return emotion_label

    if model_name == "sileod/deberta-v3-base-tasksource-nli":
        text = user_prompt
        prediction = debart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        logging.info(f"/f current classify model {model_name}, emotion label {emotion_label}")
        return emotion_label


def get_sys_prompt(user_prompt, model_name):
    emotion_label = emotion_classify(user_prompt, model_name)
    good_emotion_list = ['nature', 'happy']
    if emotion_label not in good_emotion_list:
        DEFAULT_SYS_PROMPUT = (
            f'You are a helpful assistant, you should answer the question {user_prompt} correctly,'
            f' also provide some advice for {emotion_label} emotion start with "I think you are feeling {emotion_label},'
            f'here are some suggestions for you"'
        )
        logging.info(f"current prompt {DEFAULT_SYS_PROMPUT}")
        return DEFAULT_SYS_PROMPUT
    else:
        DEFAULT_SYS_PROMPUT = f"You are a helpful assistant, you should answer the question {user_prompt} correctly."
        logging.info(f"current prompt {DEFAULT_SYS_PROMPUT}")

def generate_response(user_prompt, model_name):
    prompt = get_sys_prompt(user_prompt, model_name)
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
        temperature=0.8,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    outputs = []
    for text in streamer:
        outputs.append(text)
    result = ''.join(str(e) for e in outputs)
    return result
