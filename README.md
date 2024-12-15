# LangChain Character Information Extraction

This project uses **LangChain** and **MistralAI** to extract structured information about characters from a dataset of unstructured story files. The extracted information is output as a JSON object containing details such as the character's name, role, relationships, and a summary.

---

## **File Structure**

The project is organized as follows:

```
.
|-- cli.py                # Main CLI script to interact with the program
|-- embeddings.py         # Script for computing and storing embeddings
|-- character_extraction.py # Script for extracting structured character information
|-- .env                  # Environment file for storing API keys
|-- requirements.txt      # Dependencies required for the project
|-- README.md             # Project documentation
|-- chroma_storage/       # Directory for ChromaDB storage (auto-created)
|-- stories/              # Directory containing story files
```

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. **Install Dependencies**

Ensure you have Python 3.8+ installed. Then, install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. **Create the `.env` File**

Create a `.env` file in the root directory of the project to store your **MistralAI API key**. The `.env` file should look like this:
```bash
MISTRALAI_API_KEY=<your_api_key>
```

### 4. **Prepare the Dataset**

Download or create a directory named `stories/` in the root directory. Place your `.txt` story files inside this directory. Each file should represent a single story.

### 5. **Compute Embeddings**

Run the following command to process the stories, compute embeddings, and store them in a vector database:
```bash
python cli.py compute-embeddings --input-dir <path to stories dataset>
eg:- python cli.py compute-embeddings --input-dir D:\2024\Deepstack\data\stories
```
This command will:
- Compute embeddings for all stories in the `stories/` directory.
- Store the embeddings and metadata in **ChromaDB**.

### 6. **Extract Character Information**

To extract structured details about a specific character, run:
```bash
python cli.py get-character-info --name "<character_name>"
eg:- python cli.py get-character-info --name "Mr Holohan"
```
This will output a JSON object with the following structure:
```json
{
  "name": "Mr Holohan",
  "storyTitle": "A Mother",
  "summary": "Mr Holohan, assistant secretary of the Eire Abu Society, is arranging a series of concerts and has a game leg for which his friends call him Hoppy Holohan. He walks up and down constantly, stands by the hour at street corners arguing, and makes notes. However, it is Mrs Kearney who arranges everything in the end.",
  "relations": [],
  "characterType": "Supporting Character"
}
```

---

## **Edge Case Handling**

- If the directory specified in `compute-embeddings` does not exist, an error will be displayed.
- If the specified character is not found, the program will gracefully handle the case and return an appropriate message.

---

## **Development Notes**

- **Vector Database**: The embeddings and metadata are stored using **ChromaDB**, which persists data in the `chroma_storage/` directory.
- **LangChain Components**:
  - **Prompt Templates**: Used to instruct the language model for metadata extraction and character information.
  - **Embeddings**: Computed using `SentenceTransformer`.
  - **MistralAI**: The primary language model used for generating insights.

---

## **Dependencies**

All required dependencies are listed in the `requirements.txt` file:
- `sentence-transformers`
- `chromadb`
- `langchain-mistralai`
- `dotenv`

Install them with:
```bash
pip install -r requirements.txt
```
