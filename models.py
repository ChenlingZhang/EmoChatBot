from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login
import logging
import torch

logging.basicConfig(filename="../EmoChatBot/logs/running_log.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

login(token="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")
logging.info("Suceess login to hugging face")

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

DEFAULT_SYS_PROMPUT = """ \ You are a helpful, respectful and honest assistant. Always answer as helpfully as 
possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, 
dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a 
question does not make any sense, or is not factually coherent, explain why instead of answering something not 
correct. If you don't know the answer to a question, please don't share false information."""

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


def isCudaAvaliable() -> bool:
    if model is None:
        logging.warning("User now running on a cpu mode")
        return False
    else:
        return True


def emotion_classify(model_name, user_prompt):
    emotion_labels = ["natural", "happy", "sad", "anger", "hate"]
    if model_name == "microsoft/deberta-base-mnli":
        logging.info(f"User choose /f{model_name} as a prediction model")
        text = user_prompt
        prediction = mdebart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        return emotion_label

    if model_name == "bart-large-mnli":
        logging.info(f"User choose /f{model_name} as a prediction model")
        text = user_prompt
        prediction = bart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        return emotion_label

    if model_name == "sileod/deberta-v3-base-tasksource-nli":
        logging.info(f"User choose /f{model_name} as a prediction model")
        text = user_prompt
        prediction = debart_model_classifier(text, candidate_labels=emotion_labels)
        emotion_label = prediction['labels'][0]
        return emotion_label


def get_sys_prompt(user_prompt, model_name):
    emotion_label = emotion_classify(user_prompt, model_name)
    good_emotion_list = ['nature', 'happy']
    if emotion_label not in good_emotion_list:
        DEFAULT_SYS_PROMPUT = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as "
                               "possible,"
                               "while being safe.  Your answers should not include any harmful, unethical, racist, "
                               "sexist, toxic,"
                               "dangerous, or illegal content. Please ensure that your responses are socially "
                               "unbiased and positive in nature. If a"
                               "question does not make any sense, or is not factually coherent, explain why instead "
                               "of answering something not"
                               "correct. If you don't know the answer to a question, please don't share false "
                               "information."
                               f"also when you answer the question you should care about \f{emotion_label} emotion, "
                               f"provide some advices. here is the question\f{user_prompt}")
        logging.info("Negative Emotion, using special system prompt")
        return DEFAULT_SYS_PROMPUT
    else:
        logging.info("Positive Emotion, using default system prompt")
        DEFAULT_SYS_PROMPUT = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as "
                               "possible,"
                               "while being safe.  Your answers should not include any harmful, unethical, racist, "
                               "sexist, toxic,"
                               "dangerous, or illegal content. Please ensure that your responses are socially "
                               "unbiased and positive in nature. If a"
                               "question does not make any sense, or is not factually coherent, explain why instead "
                               "of answering something not"
                               "correct. If you don't know the answer to a question, please don't share false "
                               "information."
                               f"here is the question\f{user_prompt}")


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
