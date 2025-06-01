import chainlit as cl
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# use api key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Correct configuration for Gemini's OpenAI-compatible endpoint
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

agent1 = Agent(
    instructions="You are a helpful assistant that can answer questions and provide information.",
    name="Panaversity Support Agent"
)

@cl.on_chat_start
async def handle_chat_start():
    # Fixed: Use cl.Message instead of cl.message and send() method
    await cl.Message(content="Hello! I'm Panaversity Support Agent. How can I assist you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Process message directly without history
    result = await Runner.run(
        agent1,
        input=message.content,  # Pass actual message content
        run_config=run_config,
    )
    
    # Send response directly
    await cl.Message(content=result.final_output).send()
