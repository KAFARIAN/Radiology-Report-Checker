import os
import uuid
import torch
import operator
import traceback
import base64
from flask import send_from_directory
from werkzeug.utils import secure_filename
from typing import TypedDict, Annotated, List
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# LangChain / Ollama Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS

# Chain Imports (Standardized)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AnyMessage

# LangGraph Imports
from langgraph.graph import StateGraph, END

# App Config
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Globals
RAG_CHAIN_OBJ = None
AGENT_WORKFLOW = None
LOCAL_LLM = None
img_pipe = None

# ────────────────────────────────────────────────────────────────────────
#  File Upload Endpoint (REQUIRED FOR VISION MODEL)
# ────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        # Create a unique filename so old uploads don't get overwritten
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Return the EXACT structure your JavaScript is expecting:
        return jsonify({'message': 'File successfully uploaded', 'filepath': filepath}), 200

# ────────────────────────────────────────────────────────────────────────
#  Tool 1: Image Generation (SDXL Turbo)
# ────────────────────────────────────────────────────────────────────────
@tool
def generate_image(prompt: str) -> str:
    """
    Generate an image from a detailed text prompt using Stable Diffusion XL Turbo.
    Useful for visualizing medical concepts, anatomy, or mock radiology imaging.
    """
    global img_pipe
    if img_pipe is None:
        return "Error: Image model not loaded"

    try:
        # SDXL-Turbo usually best with 1 step + guidance=0
        image = img_pipe(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0
        ).images[0]
        
        # Make sure output dir exists
        os.makedirs('output', exist_ok=True)
        filename = f"generated_{uuid.uuid4().hex[:10]}.png"
        image_path = os.path.join('output', filename)
        image.save(image_path)

        return f"http://localhost:5000/output/{filename}"
    except Exception as e:
        return f"Image generation failed: {str(e)}"

@app.route('/output/<filename>')
def serve_generated_image(filename):
    return send_from_directory('output', filename)


# ────────────────────────────────────────────────────────────────────────
#  Tool 2: Vision Analysis (NEW - Multi-Modal)
# ────────────────────────────────────────────────────────────────────────
@tool
def analyze_radiology_image(image_path: str, query: str = "Describe this medical image in detail") -> str:
    """
    Analyzes a local image file (CT slice, X-Ray, MRI) using the Moondream Vision LLM.
    Use this tool when the user asks about a specific image file path or wants an image analyzed.
    
    Args:
        image_path: The full file path to the image on the computer.
        query: Specific question about the image (e.g., "Is there a fracture?", "Describe the density").
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    try:
        # Initialize the lightweight vision model
        # Moondream is optimized for CPU usage
        vision_llm = ChatOllama(model="moondream", temperature=0)
        
        # Read and Encode Image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Construct Multi-modal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
        )
        
        # Invoke Vision Model
        print(f"Analyzing image: {image_path} with Moondream...")
        response = vision_llm.invoke([message])
        return f"Vision Model Analysis: {response.content}"

    except Exception as e:
        traceback.print_exc()
        return f"Vision analysis failed: {str(e)}"

# ────────────────────────────────────────────────────────────────────────
#  Tool 3: RAG Knowledge Base
# ────────────────────────────────────────────────────────────────────────
def create_rag_chain_object(vectorstore):
    # Main reasoning model (Qwen 2.5)
    local_llm = ChatOllama(model="qwen2.5:3b")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    SYSTEM_PROMPT = (
        "You are WANIKO, an expert radiologist assistant. Your goal is to assist users with medical "
        "findings, reports, and visualization. "
        "\n\n"
        "INSTRUCTIONS:\n"
        "1. Accuracy: The user's question MUST be answered accurately and professionally.\n"
        "2. Tools: If the user asks to analyze an image (or uploads one), you MUST use the 'analyze_radiology_image' tool. "
        "If they ask for a visualization, use 'generate_image'.\n"
        "3. Context Awareness: You have access to the conversation history. Always check the previous messages."
        "If the user replies with short confirmations like 'Yes', 'Do it', 'Proceed', or 'Okay', "
        "you must infer their intent from the last thing you proposed (e.g., if you asked 'Shall I generate a report?', and they say 'Yes', then generate the report).\n"
        "4. Honesty: If you don't know the answer or the context is insufficient, say so clearly.\n"
        "As you ask the user for findings or medical history of a disease for the report, if the user provides the finding use, then add the findings to the reoprt.\n"
        "\n\n"
        
        "CRITICAL TOOL RULES (MANDATORY):\n"
        "1. RADIOLOGY REPORT requests = radiology_rag_tool ONLY. NEVER generate_image."
        "2. Image/analysis/visualization = generate_image or analyze_radiology_image."
        "3. Medical questions = radiology_rag_tool."

        "EXAMPLES:\n"
        "- 'Generate report' → radiology_rag_tool('chest xray pneumonia')"
        "- 'Show MRI image' → generate_image('brain MRI tumor')"
        "- 'Generate report for John Doe, Male, 45 years old with chest pain → generate_radiology_report('John Doe, Male, 45yo', 'chest pain')"
        "- 'Analyze this scan' → analyze_radiology_image(path, 'fracture?')"

        "NO IMAGES FOR REPORTS. Generate TEXT reports only."
        
        "Context from documents:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm=local_llm, prompt=prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

@tool
def radiology_rag_tool(question: str) -> str:
    """Answer questions using the radiology reports & guidelines knowledge base."""
    global RAG_CHAIN_OBJ
    if RAG_CHAIN_OBJ is None:
        return "Error: Knowledge base not initialized"

    try:
        result = RAG_CHAIN_OBJ.invoke({"input": question})
        return result["answer"]
    except Exception as e:
        return f"RAG error: {str(e)}"

@tool
def generate_radiology_report(patient_info: str, findings: str) -> str:
    """
    Generate COMPLETE radiology report.
    patient_info format: "John Doe, Male, 45 years old"
    Returns structured DICOM-standard report.
    """
    global RAG_CHAIN_OBJ
    report_prompt = f"""
    Generate PROFESSIONAL 3D CT radiology report:
    
    PATIENT: {patient_info}
    
    CLINICAL HISTORY: {findings}
    
    REQUIRED DICOM-STANDARD FORMAT:
    PATIENT: {patient_info}
    CLINICAL HISTORY: [infer from findings]
    TECHNIQUE: 3D CT [volumetric details]
    FINDINGS: [detailed observations]
    IMPRESSION: [numbered conclusions]
    
    Use radiology knowledge base ONLY.
    """
    return RAG_CHAIN_OBJ.invoke({"input": report_prompt})["answer"]


# Register all tools
AGENT_TOOLS = [radiology_rag_tool, generate_image, analyze_radiology_image, generate_radiology_report]


# ────────────────────────────────────────────────────────────────────────
#  Agent State & Graph
# ────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

def call_model(state: AgentState):
    global LOCAL_LLM
    if LOCAL_LLM is None:
        LOCAL_LLM = ChatOllama(model="qwen2.5:3b")

    # Bind tools to the LLM so it knows it can use them
    llm_with_tools = LOCAL_LLM.bind_tools(AGENT_TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_func = next((t for t in AGENT_TOOLS if t.name == tool_name), None)

        if tool_func:
            try:
                print(f"--- Agent invoking tool: {tool_name} ---")
                observation = tool_func.invoke(tool_call["args"])
            except Exception as e:
                observation = f"Tool failed: {e}"
        else:
            observation = f"Unknown tool: {tool_name}"

        results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return END

def setup_langgraph_agent(vectorstore):
    global RAG_CHAIN_OBJ, LOCAL_LLM
    LOCAL_LLM = ChatOllama(model="qwen2.5:3b")
    RAG_CHAIN_OBJ = create_rag_chain_object(vectorstore)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("call_tool", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("call_tool", "agent")
    
    return workflow.compile()


# ────────────────────────────────────────────────────────────────────────
#  Initialization Services
# ────────────────────────────────────────────────────────────────────────
def initialize_app_services():
    global AGENT_WORKFLOW, img_pipe

    # 1. Load vector store if exists
    INDEX_PATH = "radiology_faiss_index"
    
    if os.path.exists(INDEX_PATH):    
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            AGENT_WORKFLOW = setup_langgraph_agent(vectorstore)
            print(">> Agent + RAG initialized successfully")
        except Exception as e:
            print(">> Failed to initialize RAG/agent:", str(e))
            AGENT_WORKFLOW = None
    else:
        print(f"!!! WARNING: FAISS index not found at '{INDEX_PATH}'. RAG will be unavailable. !!!")
        AGENT_WORKFLOW = None

    # 2. Load SDXL-Turbo
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Path to your SDXL file
        checkpoint_path = r"C:/Users/Waniko Sebastine/Desktop/FYP/sdxl_model"

        if not os.path.exists(checkpoint_path):
             # Fallback if folder path is wrong, try to look for safetensors directly if user changed it
             print(f"Warning: SDXL Path {checkpoint_path} not found.")
        else:
            print(f"Loading SDXL-Turbo from: {checkpoint_path}")
            img_pipe = StableDiffusionXLPipeline.from_pretrained(
                checkpoint_path,
                dtype=dtype,
                safety_checker=None,
                use_safetensors=True,
                local_files_only=True
            )
            img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                img_pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            img_pipe.to(device)
            print(f">> SDXL-Turbo loaded successfully on {device}")

    except Exception as e:
        print(">> SDXL loading failed (Image gen will be disabled):", str(e))
        img_pipe = None


# ────────────────────────────────────────────────────────────────
#  Index Creation (Only runs if index is missing)
# ────────────────────────────────────────────────────────────────
def create_local_index(file_paths: list):
    print("Starting document indexing...")
    all_documents = []
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        try:
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not all_documents:
        print("No valid documents loaded → cannot create index")
        return False

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_documents)
        print(f"Created {len(chunks)} chunks. Building Embeddings (nomic-embed-text)...")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("radiology_faiss_index")
        print(f"Index saved to: radiology_faiss_index")
        return True

    except Exception as e:
        print("Index creation failed!")
        traceback.print_exc()
        return False


# ────────────────────────────────────────────────────────────────────────
#  Chat Endpoint
# ────────────────────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'reply': 'Please provide a message.'}), 400
    
    if AGENT_WORKFLOW is None:
        return jsonify({'reply': 'Error: Agent service unavailable (initialization failed).'}), 503
    
    try:
        print(f"User: {user_message}")
        result = AGENT_WORKFLOW.invoke({"messages": [HumanMessage(content=user_message)]})
        final_reply = result["messages"][-1].content
        print(f"Bot: {final_reply[:50]}...")
        return jsonify({'reply': final_reply})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'reply': f'Internal error: {str(e)}'}), 500


# ────────────────────────────────────────────────────────────────────────
# Frontend Serving Routes (Fixes 404 on http://localhost:5000)
# ────────────────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve gpt.html, gpt.css, gpt_rag.js from root folder"""
    # Map files to their names
    if path == '' or path == 'index.html':
        return send_from_directory('.', 'gpt.html')
    elif path == 'gpt.css':
        return send_from_directory('.', 'gpt.css')
    elif path == 'gpt_rag.js':
        return send_from_directory('.', 'gpt_rag.js')
    else:
        # Serve gpt.html as fallback for any other path (SPA style)
        return send_from_directory('.', 'gpt.html')

# ────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PDF_PATHS = [
        r"C:/Users/Waniko Sebastine/Desktop/FYP/pdf/Report-sample.pdf",
        r"C:/Users/Waniko Sebastine/Desktop/FYP/pdf/grainger-allisons-diagnostic-radiology-5th-edition_compress.pdf"
    ]

    INDEX_DIR = "radiology_faiss_index"
    if not os.path.exists(INDEX_DIR):
        print("Index not found → creating new one...")
        create_local_index(PDF_PATHS)
    else:
        print("Found existing FAISS index → skipping creation")

    initialize_app_services()

    print("\n" + "-"*70)
    print("Server Ready → http://localhost:5000") # http://localhost:5000
    print("Capabilities: RAG (Qwen), Vision (Moondream), Image Gen (SDXL)")
    print("-"*70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
