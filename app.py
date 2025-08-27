import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import uuid
from flask import Flask, render_template, request, redirect, session, jsonify, send_file
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import session
import tempfile
import shutil

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
api_key = os.getenv("API_KEY")

# Create a temporary directory for visualizations
VISUALIZATION_DIR = tempfile.mkdtemp(prefix='csv_visualizations_')

@app.before_request
def assign_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []

# Initialize LLM and embeddings globally (reuse)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store vector_store or dataframe and mode in session variables like keys 'vector_store' and 'df_csv' won't work (not JSON serializable)
# So we keep them as global vars keyed by session ID for simplicity in this demo
user_data = {}

def extract_text_from_pdfs(files):
    documents = []
    for f in files:
        loader = PyPDFLoader(f)
        documents.extend(loader.load())
    return documents

def create_pdf_vector_store(pdf_files):
    documents = extract_text_from_pdfs(pdf_files)
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(splitted_docs, embeddings)
    return vector_store

def create_retrieval_chain(vector_store):
    prompt_template = """
    Answer the user's question as accurately and concisely as possible based on the provided context. If not available, say you don't have the information.
    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": prompt})
    return chain

def generate_visualization(df, column_name=None, column_names=None):
    """
    Generate visualization based on column types
    Returns: base64 encoded image or None if no visualization can be generated
    """
    try:
        plt.figure(figsize=(10, 6))
        
        if column_names and len(column_names) == 2:
            # Two columns visualization
            col1, col2 = column_names
            
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Scatter plot for two numeric columns
                plt.scatter(df[col1], df[col2])
                plt.title(f'Relationship between {col1} and {col2}')
                plt.xlabel(col1)
                plt.ylabel(col2)
            elif pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
                # Box plot for numeric vs categorical
                sns.boxplot(x=col2, y=col1, data=df)
                plt.title(f'{col1} by {col2}')
                plt.xticks(rotation=45)
            elif not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Box plot for categorical vs numeric
                sns.boxplot(x=col1, y=col2, data=df)
                plt.title(f'{col2} by {col1}')
                plt.xticks(rotation=45)
            else:
                # Count plot for two categorical columns
                pd.crosstab(df[col1], df[col2]).plot(kind='bar')
                plt.title(f'Relationship between {col1} and {col2}')
                plt.xlabel(col1)
                plt.xticks(rotation=45)
                plt.legend(title=col2)
                
        elif column_name:
            # Single column visualization
            if pd.api.types.is_numeric_dtype(df[column_name]):
                # Histogram for numeric columns
                plt.hist(df[column_name].dropna(), bins=20, edgecolor='black')
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frequency')
            else:
                # Bar chart for categorical columns
                df[column_name].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
        else:
            return None
            
        plt.tight_layout()
        
        # Save plot to a file instead of base64
        img_id = str(uuid.uuid4())
        img_path = os.path.join(VISUALIZATION_DIR, f"{img_id}.png")
        plt.savefig(img_path, format='png')
        plt.close()
        
        return img_id
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        plt.close()
        return None

def enhanced_chat_csv(df, query, llm):
    """
    Enhanced CSV chat that can generate visualizations
    Returns: (answer, visualization_id) tuple
    """
    try:
        # Create a prompt that asks Gemini to suggest visualizations
        prompt_template = f"""
        You are an expert data analyst. Here is the CSV data with columns: {list(df.columns)}.
        First 5 rows:
        {df.head().to_string()}

        Based on this data, answer the question: {query}
        
        Additionally, if appropriate, suggest a visualization that would help understand the data.
        Format your response as:
        ANSWER: [your concise answer]
        VISUALIZE: [column_name] or [column1,column2] or None
        
        If a visualization would be helpful, specify the column name(s) after VISUALIZE:.
        If no visualization is needed, just write VISUALIZE: None.
        """

        response = llm.invoke(prompt_template)
        response_text = response.content
        
        # Parse the response
        answer = "No answer found"
        visualize_cols = None
        
        for line in response_text.split('\n'):
            if line.startswith('ANSWER:'):
                answer = line.replace('ANSWER:', '').strip()
            elif line.startswith('VISUALIZE:'):
                cols = line.replace('VISUALIZE:', '').strip()
                if cols.lower() != 'none':
                    visualize_cols = [c.strip() for c in cols.split(',')]
        
        # Generate visualization if suggested
        visualization_id = None
        if visualize_cols:
            if len(visualize_cols) == 1:
                visualization_id = generate_visualization(df, column_name=visualize_cols[0])
            elif len(visualize_cols) == 2:
                visualization_id = generate_visualization(df, column_names=visualize_cols)
        
        return answer, visualization_id
        
    except Exception as e:
        return f"Error: {str(e)}", None

@app.route("/", methods=["GET", "POST"])
def index():
    # Clear chat history and user data when starting a new chat
    if 'chat_history' in session:
        session.pop('chat_history')
    if session.get('user_id') in user_data:
        user_data.pop(session.get('user_id'))
    
    if request.method == "POST":
        chat_mode = request.form.get("chat_mode")
        if chat_mode == "pdf":
            pdf_files = request.files.getlist("pdf_files")
            if not pdf_files or any(f.filename == "" for f in pdf_files):
                return render_template("index.html", error="Please upload at least one PDF file.", chat_mode=chat_mode)
            # Save uploaded PDF files temporarily
            upload_dir = "uploads/pdf/"
            os.makedirs(upload_dir, exist_ok=True)
            file_paths = []
            for f in pdf_files:
                filepath = os.path.join(upload_dir, f.filename)
                f.save(filepath)
                file_paths.append(filepath)
            vector_store = create_pdf_vector_store(file_paths)
            if not vector_store:
                return render_template("index.html", error="Failed to process PDFs.", chat_mode=chat_mode)
            # Save in user_data keyed by session ID
            user_data[session['user_id']] = {"mode": "pdf", "vector_store": vector_store, "chain": create_retrieval_chain(vector_store)}
            # Add welcome message for PDF
            if 'chat_history' not in session:
                session['chat_history'] = []
            session['chat_history'].append({"role": "bot", "content": "I've processed your PDF documents. You can now ask questions about their content."})
            session.modified = True
            return redirect("/chat")
        
        elif chat_mode == "csv":
            csv_file = request.files.get("csv_file")
            if not csv_file or csv_file.filename == "":
                return render_template("index.html", error="Please upload a CSV file.", chat_mode=chat_mode)
            try:
                df = pd.read_csv(csv_file)
                user_data[session['user_id']] = {"mode": "csv", "df": df}
                # Add welcome message for CSV
                if 'chat_history' not in session:
                    session['chat_history'] = []
                columns = list(df.columns)
                sample_data = df.head(3).to_string()
                welcome_msg = f"""I've loaded your CSV data with columns: {', '.join(columns)}. Here's a sample:
                            {sample_data}

                            💬 Gemini-Powered CSV Chat
                            Try questions like:
                            - "show me the correlation between column_name1 and column_name2"
                            - "Show relationship between column_name1 and column_name2"
                            - "What is the distribution of column_name1?"
                            - "Compare column_name1 by column_name2"
                            - "What is the maximum column_name1 from this csv?"

                            You can now ask questions about this data."""
                session['chat_history'].append({"role": "bot", "content": welcome_msg})
                session.modified = True
                return redirect("/chat")
            except Exception as e:
                return render_template("index.html", error=f"Error reading CSV file: {str(e)}", chat_mode=chat_mode)
        else:
            return render_template("index.html", error="Invalid chat mode selected.")
    else:
        return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_session = user_data.get(session['user_id'])
    if not user_session:
        return redirect("/")
    
    mode = user_session["mode"]
    
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == "POST":
        user_query = request.form.get("query")
        if not user_query or user_query.strip() == "":
            # If empty query, just redisplay the chat
            return render_template("chat.html", mode=mode, chat_history=session['chat_history'])
        
        # Add user message to chat history
        session['chat_history'].append({"role": "user", "content": user_query})
        
        try:
            if mode == "pdf":
                chain = user_session.get("chain")
                if not chain:
                    return redirect("/")
                result = chain.invoke({"query": user_query})
                response_text = result["result"]
                # For PDF mode, no visualization
                visualization_id = None
                
            elif mode == "csv":
                df = user_session.get("df")
                if df is None:
                    return redirect("/")
                # Use the enhanced chat function
                response_text, visualization_id = enhanced_chat_csv(df, user_query, llm)
                
            # Add bot response to chat history
            session['chat_history'].append({
                "role": "bot", 
                "content": response_text,
                "visualization": visualization_id
            })
            
        except Exception as e:
            # Add error message to chat history
            error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
            session['chat_history'].append({"role": "bot", "content": error_msg})
            print(f"Error: {str(e)}")
        
        # Save the session
        session.modified = True
        
        return render_template("chat.html", mode=mode, chat_history=session['chat_history'])
    
    return render_template("chat.html", mode=mode, chat_history=session['chat_history'])

# Add a new route to serve visualization images
@app.route("/visualization/<visualization_id>")
def get_visualization(visualization_id):
    img_path = os.path.join(VISUALIZATION_DIR, f"{visualization_id}.png")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    return "Visualization not found", 404

# Clean up visualization files when the app shuts down
import atexit
def cleanup_visualizations():
    if os.path.exists(VISUALIZATION_DIR):
        shutil.rmtree(VISUALIZATION_DIR)
atexit.register(cleanup_visualizations)

if __name__ == "__main__":
    app.run(debug=True)