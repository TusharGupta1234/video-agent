# Actionable items, decisions, questions
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def extract_with_chunking(transcript: str, map_system_prompt: str, combine_system_prompt: str) -> str:
    llm = get_llm()

    # 1. Map Step: Run extraction on individual chunks
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", map_system_prompt),
        ("human", "{text}"),
    ])
    
    map_chain = map_prompt | llm | StrOutputParser()
    
    chunks = split_transcript(transcript)
    chunk_extractions = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    
    # 2. Combine Step: Consolidate partial extractions
    combined_text = "\n\n--- Next Chunk Extraction ---\n\n".join(chunk_extractions)
    
    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", combine_system_prompt),
        ("human", "{text}"),
    ])
    
    combine_chain = (
        RunnablePassthrough() 
        | RunnableLambda(lambda x: {"text": x}) 
        | combine_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return combine_chain.invoke(combined_text)


def extract_action_items(transcript: str) -> str:
    map_prompt = (
        "You are an expert meeting analyst. From this portion of the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    
    combine_prompt = (
        "You are an expert meeting analyst. I am providing you with action items extracted "
        "from various chunks of a single meeting transcript. Your job is to consolidate them "
        "into one final, deduplicated numbered list. Maintain this format for each:\n"
        "- Task description\n"
        "- Owner\n"
        "- Deadline\n\n"
        "If no action items were found across all chunks, say 'No action items found.'"
    )
    
    return extract_with_chunking(transcript, map_prompt, combine_prompt)


def extract_key_decisions(transcript: str) -> str:
    map_prompt = (
        "You are an expert meeting analyst. From this portion of the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    
    combine_prompt = (
        "You are an expert meeting analyst. Combine these key decisions extracted from "
        "different parts of a meeting into a single, comprehensive, deduplicated numbered list. "
        "If no key decisions were found across all chunks, say 'No key decisions found.'"
    )
    
    return extract_with_chunking(transcript, map_prompt, combine_prompt)


def extract_questions(transcript: str) -> str:
    map_prompt = (
        "From this portion of the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    
    combine_prompt = (
        "Combine these unresolved questions extracted from different parts of a meeting "
        "into a single, comprehensive, deduplicated numbered list. "
        "If no open questions were found across all chunks, say 'No open questions found.'"
    )
    
    return extract_with_chunking(transcript, map_prompt, combine_prompt)