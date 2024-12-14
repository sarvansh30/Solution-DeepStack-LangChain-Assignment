import os
import argparse
import json
from dotenv import load_dotenv

from embeddings import compute_story_embeddings
from character_extraction import extract_character_info

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="LangChain Character Information Extraction")
    subparsers = parser.add_subparsers(dest='command', help='CLI Commands')

    embedding_parser = subparsers.add_parser('compute-embeddings', help='Compute embeddings for stories')
    embedding_parser.add_argument('--input-dir', 
                                  required=True, 
                                  help='Directory containing story files')

    character_parser = subparsers.add_parser('get-character-info', 
                                             help='Extract structured information about a character')
    character_parser.add_argument('--name', 
                                  required=True, 
                                  help='Name of the character to extract information about')

    args = parser.parse_args()

    if args.command == 'compute-embeddings':
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory {args.input_dir} does not exist.")
            return
        
        print(f"Computing embeddings for stories in {args.input_dir}")
        compute_story_embeddings(args.input_dir)
        print("Embeddings computation completed.")

    elif args.command == 'get-character-info':
        try:
            character_info = extract_character_info(args.name)
            
        
            print(json.dumps(character_info, indent=2))
        
        except ValueError as e:
            print(f"Error: {e}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()