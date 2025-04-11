import chainlit as cl
import asyncio
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from textwrap import dedent

# LLMs
llm_base = OllamaLLM(model="ollama/mistral")
llm_streaming = OllamaLLM(model="mistral", streaming=True)

# Streaming helper
async def stream_response(agent_name: str, prompt: str):
    msg = cl.Message(content="", author=f"{agent_name}")
    await msg.send()

    full_text = ""
    async for token in llm_streaming.astream(prompt):
        full_text += token
        await msg.stream_token(token)

    msg.content = full_text.strip()
    await msg.update()
    return full_text.strip()

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    await cl.Message(content=f"❓ Question: {question}").send()

    memory = {}

    # --- Agents ---
    rephraser = Agent(
        role="Rephraser",
        goal="Clarify vague or ambiguous user questions",
        backstory="An expert in user intent disambiguation and rephrasing.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    answerer = Agent(
        role="Answer Generator",
        goal="Provide a comprehensive and thoughtful answer",
        backstory="A deep reasoning expert trained on a vast corpus of knowledge.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal="Ensure factual accuracy and provide credible sources",
        backstory="An information analyst with strong attention to accuracy.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    summarizer = Agent(
        role="Summarizer",
        goal="Summarize the conversation in a useful, informative way",
        backstory="A master at synthesis and TL;DRs.",
        verbose=True,
        allow_delegation=False,
        llm=llm_base,
    )

    # --- Tasks ---
    task_rephrase = Task(
        description=f"Rephrase the following user question for clarity:\n\n{question}",
        agent=rephraser,
        expected_output="A clearer version of the question for AI understanding.",
    )

    task_answer = Task(
        description=dedent(f"""
        Based on the rephrased question, give a detailed and thoughtful answer.
        Wait for the Rephraser agent's output first.
        """),
        agent=answerer,
        expected_output="A thorough answer to the rephrased question.",
        depends_on=[task_rephrase],
    )

    task_factcheck = Task(
        description=dedent(f"""
        Critically verify the answer generated and cite any sources you can infer.
        Respond with a clear verdict and references.
        """),
        agent=fact_checker,
        expected_output="A fact-check summary with evidence.",
        depends_on=[task_answer],
    )

    task_summary = Task(
        description=dedent(f"""
        Combine the answer and fact-checking into a short and useful summary.
        """),
        agent=summarizer,
        expected_output="A TL;DR-style summary.",
        depends_on=[task_factcheck],
    )

    # Crew config
    crew = Crew(
        agents=[rephraser, answerer, fact_checker, summarizer],
        tasks=[task_rephrase, task_answer, task_factcheck, task_summary],
        verbose=False
    )

    # Execute CrewAI
    result = await asyncio.to_thread(crew.kickoff)

    # --- Stream Each Agent's Result ---
    memory["🔄 Reformulated"] = task_rephrase.output.raw
    memory["📘 Answer"] = task_answer.output.raw
    memory["🔍 Fact Check"] = task_factcheck.output.raw
    memory["📝 Summary"] = task_summary.output.raw

    for label, content in memory.items():
        await stream_response(label, content)
