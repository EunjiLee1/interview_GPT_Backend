import gradio as gr
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessage,
    HumanMessage
)
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from omegaconf import OmegaConf
import pandas as pd

conf = OmegaConf.load('app/prompt/config.yaml')
interview = conf.interview

prompt_path = 'app/prompt/prompt.yaml'
prompt = OmegaConf.load(prompt_path)
summarize_sys_template = prompt.summarize.system_prompt
summarize_user_template = prompt.summarize.user_prompt

sum_sys_message_prompt = SystemMessagePromptTemplate.from_template(summarize_sys_template)
sum_human_message_prompt = HumanMessagePromptTemplate.from_template(summarize_user_template)
sum_messages = ChatPromptTemplate.from_messages([sum_sys_message_prompt, sum_human_message_prompt])

LLM = OpenAI(
    openai_api_key='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',
    temperature=prompt.chat.temperature
    # max_tokens=prompt.chat.max_tokens,
    # model_kwargs={'top_p':conf.chat.top_p}
)

sum_prompt = sum_messages.format_prompt(interview=interview)
sum_prompt_str = sum_prompt.to_string()

msg = LLM(sum_prompt_str)

print(sum_prompt_str)
print(msg)