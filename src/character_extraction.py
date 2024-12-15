import chromadb
import difflib
from sentence_transformers import SentenceTransformer
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class CharacterExtractor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        self.collection = self.chroma_client.get_or_create_collection(name="story_embeddings")
        self.llm = ChatMistralAI()
        self.character_prompt = ChatPromptTemplate.from_messages([
            ("system", """Carefully extract structured character information from the story. 
            Only return information if the character is DEFINITIVELY present in the story.
            Provide a JSON object with the following keys:
            - name: Character's full name (MUST exactly match the input name)
            - storyTitle: Title of the story
            - summary: Brief character summary
            - relations: List of character relationships
            - characterType: Role in the story

            If the character is NOT clearly present, return only character name and status: NOT present in any stories."""),
            ("human", "Character Name: {character_name}\n\nStory Context: {story_context}")
        ])
        self.json_parser = JsonOutputParser()
        self.extraction_chain = self.character_prompt | self.llm | self.json_parser

    def _is_name_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return True
        
        similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        
        return similarity >= threshold

    def extract_character_info(self, character_name: str):
        if not character_name or len(character_name.strip()) < 2:
            raise ValueError("Character name must be at least 2 characters long")
        
        normalized_name = character_name.strip()
        character_embedding = self.embedding_model.encode(normalized_name).tolist()
       
        search_results = self.collection.query(
            query_embeddings=[character_embedding],
            n_results=3 
        )
     
        if not search_results['documents'] or len(search_results['documents'][0]) == 0:
            raise ValueError(f"No stories found containing information about {normalized_name}")
     
        for story_text, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
            try:
                character_info = self.extraction_chain.invoke({
                    "character_name": normalized_name,
                    "story_context": story_text
                })
                
                if character_info.get('name') and self._is_name_similar(character_info['name'], normalized_name):
                    return character_info
            
            except Exception as e:
                pass
        
        raise ValueError(f"Could not find definitive information for {normalized_name}")

def extract_character_info(character_name: str):
    extractor = CharacterExtractor()
    return extractor.extract_character_info(character_name)