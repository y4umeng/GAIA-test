from transformers.agents.llm_engine import MessageRole, get_clean_message_list
import os
from openai import OpenAI
from anthropic import Anthropic, AnthropicBedrock
import ollama
openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[], grammar=None):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
            response_format=grammar
        )
        return response.choices[0].message.content


class AnthropicEngine:
    def __init__(self, model_name="claude-3-5-sonnet-20240620", use_bedrock=False):
        self.model_name = model_name
        if use_bedrock: # Cf this page: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
            self.model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            self.client = AnthropicBedrock(
                aws_access_key=os.getenv("AWS_BEDROCK_ID"),
                aws_secret_key=os.getenv("AWS_BEDROCK_KEY"),
                aws_region="us-east-1",
            )
        else:
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        index_system_message, system_prompt = None, None
        for index, message in enumerate(messages):
            if message["role"] == MessageRole.SYSTEM:
                index_system_message = index
                system_prompt = message["content"]
        if system_prompt is None:
            raise Exception("No system prompt found!")

        filtered_messages = [message for i, message in enumerate(messages) if i!= index_system_message]
        if len(filtered_messages) == 0:
            print("Error, no user message:", messages)
            assert False

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=filtered_messages,
            stop_sequences=stop_sequences,
            temperature=0.5,
            max_tokens=2000
        )
        full_response_text = ""
        for content_block in response.content:
            if content_block.type == 'text':
                full_response_text += content_block.text
        return full_response_text

class OllamaEngine:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

    def __call__(self, messages, stop_sequences=[], grammar=None, temp=0.5):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        print(f"~~~~~Calling OLLAMA~~~~~")
        print(messages)
        print("\nSTOP SEQ:")
        print(stop_sequences)
        print("~~~~~~~~~~~~~")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=temp, # https://github.com/ollama/ollama/issues/6640#issuecomment-2330311462
            response_format=grammar
        )
        print("RESULT::")
        print(response.choices[0].message.content)
        print("~~~~END RESULT~~~~~~")
        return response.choices[0].message.content

llama_role_conversions = openai_role_conversions

class OllamaEngine2:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.client = ollama

    def __call__(self, messages, stop_sequences=[]) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get LLM output
        response = self.client.chat(model=self.model_name, messages=messages)['message']['content']
        # print(f"OUTPUT PRE PARSE:\n\n{response}\n\n")
        # Remove stop sequences from LLM output
        for stop_seq in stop_sequences:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        # print(f"OUTPUT POST PARSE:\n\n{response}\n\n")
        # if not response: print(f"WHERES THE RESPONSE?? MESSAGES:\n\n{messages}\n\n")
        return response

# import boto3
# import json
# import os

# class AnthropicBedrockEngine:
#     def __init__(self, model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"):
#         self.model_name = model_name
#         self.bedrock_runtime = boto3.client(
#             service_name='bedrock-runtime',
#             region_name='us-east-1',
#             aws_access_key_id=os.getenv("AWS_BEDROCK_ID"),
#             aws_secret_access_key=os.getenv("AWS_BEDROCK_KEY")
#         )


#     def __call__(self, messages, stop_sequences=[]):
#         messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
#         index_system_message, system_prompt = None, None
#         for index, message in enumerate(messages):
#             if message["role"] == MessageRole.SYSTEM:
#                 index_system_message = index
#                 system_prompt = message["content"]
#         if system_prompt is None:
#             raise Exception("No system prompt found!")

#         filtered_messages = [message for i, message in enumerate(messages) if i!= index_system_message]
#         if len(filtered_messages) == 0:
#             print("Error, no user message:", messages)
#             assert False

#         request = json.dumps({
#             "anthropic_version": "bedrock-2023-05-31",    
#             "max_tokens": 2000,
#             "system": system_prompt,    
#             "messages": filtered_messages,
#             "temperature": 0.5,
#             "stop_sequences": stop_sequences
#         })
#         response = self.bedrock_runtime.invoke_model(body=request, modelId=self.model_name)
#         response_body = json.loads(response.get('body').read())
    
#         full_response_text = ""
#         return full_response_text