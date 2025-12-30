import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings()

with open("data.json", "r") as f:
    data = json.load(f)

anime_docs = []
for exp in data["anime_experiences"]:
    content = f"""Title: {exp['title']}
Rating: {exp['rating']}/10
Year Watched: {exp['year_watched']}
Favorite Character: {exp['favorite_character']}
Experience: {exp['experience']}
Thoughts: {exp['thoughts']}"""
    anime_docs.append(Document(
        page_content=content,
        metadata={"type": "anime", "title": exp['title'], "rating": exp['rating']}
    ))

game_docs = []
for exp in data["game_experiences"]:
    content = f"""Title: {exp['title']}
Rating: {exp['rating']}/10
Year Played: {exp['year_played']}
Platform: {exp['platform']}
Playtime: {exp['playtime_hours']} hours
Experience: {exp['experience']}
Thoughts: {exp['thoughts']}"""
    game_docs.append(Document(
        page_content=content,
        metadata={"type": "game", "title": exp['title'], "rating": exp['rating']}
    ))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

anime_splits = text_splitter.split_documents(anime_docs)
game_splits = text_splitter.split_documents(game_docs)

print("Creating anime vector store...")
anime_vectorstore = Chroma.from_documents(
    documents=anime_splits,
    embedding=embeddings,
    collection_name="anubhav_anime_experiences",
    persist_directory="./chroma_db/anime"
)

print("Creating game vector store...")
game_vectorstore = Chroma.from_documents(
    documents=game_splits,
    embedding=embeddings,
    collection_name="anubhav_game_experiences",
    persist_directory="./chroma_db/games"
)

print("Vector databases created successfully!")
print(f"   - Anime: {len(anime_splits)} chunks stored")
print(f"   - Games: {len(game_splits)} chunks stored")
