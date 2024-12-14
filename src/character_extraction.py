import chromadb
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
            ("system", """Extract structured character information from the story. 
            Provide a JSON object with the following keys:
            - name: Character's full name
            - storyTitle: Title of the story
            - summary: Brief character summary
            - relations: List of character relationships
            - characterType: Role in the story

            If the character is not found, return an empty object."""),
            ("human", "Character Name: {character_name}\n\nStory Context: {story_context}")
        ])

        self.json_parser = JsonOutputParser()
       
        self.extraction_chain = self.character_prompt | self.llm | self.json_parser

    def extract_character_info(self, character_name: str):
        
        character_embedding = self.embedding_model.encode(character_name).tolist()
       
        search_results = self.collection.query(
            query_embeddings=[character_embedding],
            n_results=3 
        )
     
        if not search_results['documents'] or len(search_results['documents'][0]) == 0:
            raise ValueError(f"No stories found containing information about {character_name}")
     
        for story_text, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
            try:
                character_info = self.extraction_chain.invoke({
                    "character_name": character_name,
                    "story_context": story_text
                })
            
                if character_info.get('name'):
                    return character_info
            
            except Exception as e:
                print(f"Error processing story: {e}")
        
        # If no character information found
        raise ValueError(f"Could not find detailed information for {character_name}")

def extract_character_info(character_name: str):

    extractor = CharacterExtractor()
    return extractor.extract_character_info(character_name)