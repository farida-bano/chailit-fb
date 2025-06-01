import chainlit as cl
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
load_dotenv(find_dotenv())

# Get API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Validate API key exists
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. "
                     "Please set it in your .env file or environment variables.")

# Correct Gemini configuration
provider = AsyncOpenAI(
    api_key=gemini_api_key,  # Use the variable here
    base_url="https://generativelanguage.googleapis.com/v1beta",  # Correct endpoint
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
    await cl.Message(content="Hello! I'm Panaversity Support Agent. How can I assist you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(
        agent1,
        input=message.content,
        run_config=run_config,
    )
    await cl.Message(content=result.final_output).send()