import chainlit as cl
import asyncio
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from textwrap import dedent

# 1. LLMs
llm_base = OllamaLLM(model="ollama/mistral")
llm_streaming = OllamaLLM(model="mistral", streaming=True)


# 2. Helper to send messages
# async def send_msg(label: str, content: str, author: str = "🤖 Agent"):
#     await cl.Message(content=f"{label} {content}", author=author).send()

# Helper to stream messages
async def stream_response(agent_name: str, prompt: str):
    msg = cl.Message(content="", author=f"📘 {agent_name}")
    await msg.send()

    full_text = ""
    async for token in llm_streaming.astream(prompt):
        full_text += token
        await msg.stream_token(token)

    msg.content = full_text.strip()
    await msg.update()
    return full_text.strip()


# 3. Chainlit + CrewAI Integration
@cl.on_message
async def on_message(message: cl.Message):
    question = message.content
    await cl.Message(content=f"❓ Question: {question}").send()

    # Define Agents
    reformulate_agent = Agent(
        role="Rephraser",
        goal="Clarify user input for better understanding",
        backstory="Expert in NLP and user intent analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    answer_agent = Agent(
        role="Answer Generator",
        goal="Generate clear and helpful answers to user questions.",
        backstory="Trained on vast knowledge to answer accurately.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    fact_check_agent = Agent(
        role="Fact Checker",
        goal="Verify facts and cite sources when possible.",
        backstory="Expert researcher with access to verified knowledge.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    summary_agent = Agent(
        role="Summarizer",
        goal="Summarize key points including fact-checks.",
        backstory="Summarizes answers with concise language.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    # Define Tasks
    reformulate_task = Task(
        description=f"Rephrase this question clearly: {question}",
        agent=reformulate_agent,
        expected_output="A clearly reworded question.",
    )

    answer_task = Task(
        description="Generate a helpful and detailed answer to the question.",
        agent=answer_agent,
        expected_output="A helpful answer to the question.",
        depends_on=[reformulate_task],
    )

    fact_check_task = Task(
        description=dedent("""
            Verify the accuracy of the answer and cite any relevant sources.
        """),
        agent=fact_check_agent,
        expected_output="A verdict on accuracy with possible citations.",
        depends_on=[answer_task],
    )

    summary_task = Task(
        description=dedent("""
            Summarize the answer and include fact-check results.
        """),
        agent=summary_agent,
        expected_output="A concise summary with fact-check insights.",
        depends_on=[fact_check_task],
    )

    # Create Crew
    crew = Crew(
        agents=[reformulate_agent, answer_agent, fact_check_agent, summary_agent],
        tasks=[reformulate_task, answer_task, fact_check_task, summary_task],
        verbose=False
    )

    # Run CrewAI in a separate thread to avoid blocking Chainlit
    result = await asyncio.to_thread(crew.kickoff)

    # Send results stage by stage
    await stream_response("🔄 Reformulated", reformulate_task.output.raw)
    await stream_response("📘 Answer", answer_task.output.raw)
    await stream_response("🔍 Fact Check", fact_check_task.output.raw)
    await stream_response("📝 Summary", summary_task.output.raw)

#  chainlit run /Users/dhananjayasamantasinghar/Desktop/test-python/src/test/test_pyspark/crewai_chatbot.py
