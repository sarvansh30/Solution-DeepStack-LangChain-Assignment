import os
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class StoryEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        self.collection = self.chroma_client.get_or_create_collection(name="story_embeddings")
        
        self.llm = ChatMistralAI()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key metadata from the following story:"),
            ("human", "{story}")
        ])
        
        self.output_parser = StrOutputParser()
        
        self.chain = self.prompt | self.llm | self.output_parser

    def _read_story(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def compute_embeddings(self, stories_dir: str):
        for filename in os.listdir(stories_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(stories_dir, filename)
                story_text = self._read_story(file_path)
                
                try:
                    metadata_str = self.chain.invoke({"story": story_text})
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

                embedding = self.embedding_model.encode(story_text).tolist()

                self.collection.add(
                    embeddings=[embedding],
                    documents=[story_text],
                    metadatas=[{"filename": filename, "metadata": metadata_str}],
                    ids=[filename]
                )
        
        print("Embeddings computed and stored successfully.")

def compute_story_embeddings(stories_dir: str):
    """Wrapper function to compute embeddings."""
    embedder = StoryEmbedder()
    embedder.compute_embeddings(stories_dir)