const codeOptions = {
    type1: `# Python code for Question and answering chatbot\n
pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain\n`,
    type2: `# Python code for Semantic search engine\n
pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain\n`,    
    splitter1: `\n# Python code for Character Text Splitter\nfrom langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
separator = '\\n\\n', 
# It is recommended to have chunk overlap between 0-25% of chunk size.
chunk_size = {{chunkSize}},
chunk_overlap  = {{chunkOverlap}},
length_function = len,
)`,
    splitter2: `\n# Python code for Tiktoken Text Splitter\nfrom langchain.text_splitter import TokenTextSplitter
# It is recommended to have chunk overlap between 0-25% of chunk size.
text_splitter = TokenTextSplitter(chunk_size={{chunkSize}}, chunk_overlap={{chunkOverlap}})`,

    notionFiles: `\n# Python code for Importing Notion Files \nfrom langchain.document_loaders import NotionDirectoryLoader\nloader_notion = NotionDirectoryLoader("Notion_DB") #Insert Notion db path here\ndocs_notion = loader_notion .load()`,
    pdfFiles: `\n# Python code for Importing PDF Files \nfrom langchain.document_loaders import PyPDFLoader\nloader_pdf = PyPDFLoader("example_data/layout-parser-paper.pdf") #Insert pdf path here\ndocs_pdf = loader_pdf .load()`,
    wordFiles: `\n# Python code for Importing Word Files \nfrom langchain.document_loaders import Docx2txtLoader\nloader_word = Docx2txtLoader("example_data/fake.docx") #Insert word doc path here\ndocs_word = loader_word .load()`,
    csvFiles: `\n# Python code for Importing CSV Files \nfrom langchain.document_loaders.csv_loader import CSVLoader\nloader_csv = CSVLoader(file_path='./example_data/mlb_teams_2012.csv')\ndata_csv = loader_csv.load()`,
    chromaDB: {
        yes: `\n# Python code for ChromaDB with saving embeddings\npersist_directory = 'db' #This is the embeddings directory\nvectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)`,
        no: `\n# Python code for ChromaDB without saving embeddings\ndb = Chroma.from_documents(docs, embeddings)`
    },
    pinecone: `\n# Python code for Pinecone\n# Add your Pinecone related code here...`
};


const defaultValues = {
    splitter1: { chunkSize: 1000, chunkOverlap: 200 },
    splitter2: { chunkSize: 100, chunkOverlap: 0 },
};


document.getElementById("chatbotType").addEventListener("change", updateCode);
document.getElementById("splitterType").addEventListener("change", function() {
    updateDefaults();    
    updateCode();
});
document.getElementById("chunkSize").addEventListener("change", updateCode);
document.getElementById("chunkOverlap").addEventListener("change", updateCode);
document.getElementById("notionFiles").addEventListener("change", updateCode);
document.getElementById("pdfFiles").addEventListener("change", updateCode);
document.getElementById("wordFiles").addEventListener("change", updateCode);
document.getElementById("csvFiles").addEventListener("change", updateCode);

function updateCode() {
    let selectedType = document.getElementById("chatbotType").value;
    let selectedSplitter = document.getElementById("splitterType").value;
    let chunkSize = document.getElementById("chunkSize").value;
    let chunkOverlap = document.getElementById("chunkOverlap").value;
    let codeDisplay = document.getElementById("codeDisplay");

    let codeToDisplay = codeOptions[selectedType] + "\n" + codeOptions[selectedSplitter];

    // Replace placeholders in the code with the actual values
    codeToDisplay = codeToDisplay.replace("{{chunkSize}}", chunkSize);
    codeToDisplay = codeToDisplay.replace("{{chunkOverlap}}", chunkOverlap);

    if (document.getElementById("notionFiles").checked) {
        codeToDisplay += "\n" + codeOptions["notionFiles"];
    }
    if (document.getElementById("pdfFiles").checked) {
        codeToDisplay += "\n" + codeOptions["pdfFiles"];
    }
    if (document.getElementById("wordFiles").checked) {
        codeToDisplay += "\n" + codeOptions["wordFiles"];
    }
    if (document.getElementById("csvFiles").checked) {
        codeToDisplay += "\n" + codeOptions["csvFiles"];
    }

    let selectedDB = document.getElementById("dbType").value;
    let saveEmbeddings = document.querySelector('input[name="saveEmbeddings"]:checked')?.value;
    
    if (selectedDB) {
        if (selectedDB === "chromaDB" && saveEmbeddings) {
            codeToDisplay += "\n" + codeOptions[selectedDB][saveEmbeddings];
        } else if (selectedDB === "pinecone") {
            codeToDisplay += "\n" + codeOptions[selectedDB];
        }
    }

    codeDisplay.textContent = codeToDisplay;

    // Highlight the newly inserted code
    Prism.highlightElement(codeDisplay);
}

function updateDefaults() {
    let selectedSplitter = document.getElementById("splitterType").value;
    let chunkSize = document.getElementById("chunkSize");
    let chunkOverlap = document.getElementById("chunkOverlap");

    chunkSize.value = defaultValues[selectedSplitter].chunkSize;
    chunkOverlap.value = defaultValues[selectedSplitter].chunkOverlap;
}


document.getElementById("chunkSize").addEventListener("input", checkChunkSize);
document.getElementById("chunkOverlap").addEventListener("input", checkChunkSize);

function checkChunkSize() {
    let chunkSize = parseInt(document.getElementById("chunkSize").value);
    let chunkOverlap = parseInt(document.getElementById("chunkOverlap").value);
    let warningMessage = document.getElementById("warningMessage");

    if (chunkOverlap >= chunkSize) {
        warningMessage.style.display = "block";
    } else {
        warningMessage.style.display = "none";
    }
}

document.getElementById("dbType").addEventListener("change", function() {
    updateCode();
    updateSaveEmbeddingsSection();
});
document.getElementById("saveEmbeddingsYes").addEventListener("change", updateCode);
document.getElementById("saveEmbeddingsNo").addEventListener("change", updateCode);

function updateSaveEmbeddingsSection() {
    let selectedDB = document.getElementById("dbType").value;
    let saveEmbeddingsSection = document.getElementById("saveEmbeddingsSection");

    if (selectedDB === "chromaDB") {
        saveEmbeddingsSection.style.display = "block";
    } else {
        saveEmbeddingsSection.style.display = "none";
    }
}


document.getElementById("copyButton").addEventListener("click", function() {
    // Create a temporary textarea to hold the code
    let tempTextArea = document.createElement("textarea");

    // Set the value of the textarea to the code
    tempTextArea.value = document.getElementById("codeDisplay").textContent;

    // Append the textarea to the body
    document.body.appendChild(tempTextArea);

    // Select the text in the textarea
    tempTextArea.select();

    // Copy the text to the clipboard
    document.execCommand("copy");

    // Remove the textarea from the body
    document.body.removeChild(tempTextArea);

    // Alert the user that the code has been copied
    alert("Code copied to clipboard!");
});


// Call updateDefaults initially to set the default values based on the initially selected splitter type
updateDefaults();