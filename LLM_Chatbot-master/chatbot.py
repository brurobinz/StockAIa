import os
import re
import logging
import re
import torch
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from dateutil import parser  # For flexible date parsing
from transformers import BitsAndBytesConfig
# ------------------------------
# 1. Configuration and Setup
# ------------------------------

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable.")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Pinecone client: {e}")
    raise

index_name = 'thesis-database2'

# Check if the index exists; if not, create it
try:
    existing_indexes = [index.name for index in pc.list_indexes().indexes]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # Ensure this matches your embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Index '{index_name}' created successfully.")
    else:
        logger.info(f"Index '{index_name}' already exists.")
except Exception as e:
    logger.error(f"Error checking or creating index '{index_name}': {e}")
    raise

# Connect to the index
try:
    index = pc.Index(index_name)
    logger.info(f"Connected to Pinecone index '{index_name}' successfully.")
except Exception as e:
    logger.error(f"Error connecting to Pinecone index '{index_name}': {e}")
    raise

# Initialize the language model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HUGGINGFACE_API_TOKEN,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HUGGINGFACE_API_TOKEN,
        torch_dtype=torch.float32,  # CPU
        device_map="auto",
        trust_remote_code=True
    )
    logger.info(f"Loaded model '{model_name}'.")
except Exception as e:
    logger.exception(f" Failed to load model '{model_name}'")
    tokenizer = None
    model = None

# Ensure pad_token_id and eos_token_id are defined
if tokenizer and model:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')

# Initialize the SentenceTransformer model for embeddings
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token="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")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = embedding_model.to(device)
    logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2' model.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise

# Mapping from query price types to metadata fields
PRICE_TYPE_MAPPING = {
    'opening': 'open',
    'closing': 'close',
    'high': 'high',
    'low': 'low',
    'open': 'open',
    'close': 'close',
    'high price': 'high',
    'low price': 'low'
}

# ------------------------------
# 2. Helper Functions
# ------------------------------

def construct_prompt(query, matches):
    """
    Constructs a concise and clear prompt for the language model using the retrieved context.
    """

    # Organize context by type
    context_sections = {
        'stock': [],
        'news': [],
        'tweet': [],
        'forecast': []
    }

    for match in matches:
        metadata = match['metadata']
        if metadata.get('type') == 'stock':
            symbol = metadata.get('symbol', 'Unknown Symbol')
            date = metadata.get('date', 'Unknown Date')
            open_price = metadata.get('open', 'Unknown Open')
            close_price = metadata.get('close', 'Unknown Close')
            high_price = metadata.get('high', 'Unknown High')
            low_price = metadata.get('low', 'Unknown Low')
            summary = metadata.get('summary', '')
            stock_info = (
                f"- Symbol: {symbol}, Date: {date}\n"
                f"  Open: ${open_price}, Close: ${close_price}, High: ${high_price}, Low: ${low_price}\n"
                f"  Summary: {summary}"
            )
            context_sections['stock'].append(stock_info)

        elif metadata.get('type') in ['news', 'news2']:
            headline = metadata.get('headline', 'No Headline')
            content = metadata.get('content', 'No Content')
            publication_date = metadata.get('publication_date', 'Unknown Date')
            source_url = metadata.get('source_url', 'No Source URL')
            news_info = (
                f"- Headline: {headline} ({publication_date})\n"
                f"  Content: {content}\n"
                f"  Source: {source_url}"
            )
            context_sections['news'].append(news_info)

        elif metadata.get('type') == 'tweet':
            writer = metadata.get('writer', 'Unknown')
            post_date = metadata.get('post_date', 'Unknown Date')
            text = metadata.get('text', '')
            tweet_info = f"- {writer} on {post_date}: {text}"
            context_sections['tweet'].append(tweet_info)

        elif metadata.get('type') == 'forecast':
            symbol = metadata.get('symbol', 'Unknown Symbol')
            date = metadata.get('date', 'Unknown Date')
            predicted_price = metadata.get('predicted_price', 'Unknown')
            forecast_info = f"- Symbol: {symbol}, Date: {date}, Predicted Price: ${predicted_price}"
            context_sections['forecast'].append(forecast_info)

    # Build full context string
    context = ""
    for section, items in context_sections.items():
        if items:
            context += f"\n{section.capitalize()} Information:\n"
            context += "\n".join(items) + "\n"

    # Final prompt construction
    prompt = f"""
You are a helpful and unbiased financial assistant.

Your job is to answer the user's question clearly and briefly, based only on the information provided in the context.

Instructions:
- Always start your answer with "Yes." or "No." — do not explain first.
- Follow with 1–2 short, non-repetitive reasons based only on the context.
- Do not repeat the same idea in different words.
- Do not guess or invent information.
- If the context lacks relevant data, say: "There is not enough information to answer confidently."
- If the user asks for profit from 1000 shares of COST in 2 years, and forecast is available, show clear steps and final profit.

Examples:

Q: Should I invest in XYZ now?  
A: No. The company reported a 20% drop in revenue last quarter, and recent news suggests weak future growth.

Q: Is ABC a good investment?  
A: Yes. ABC has shown strong earnings growth recently and has outperformed competitors.

Context:
{context.strip()}

Question:
{query}

Answer:
""".strip()

    return prompt


def parse_stock_query(query):
    """
    Parses the user's query to extract price type, stock symbol, and date.
    Supports both regular and forecast-related queries.
    Returns a tuple: (price_type, symbol, date)
    """
    # Existing price query patterns
    price_patterns = [
        r"What was the (\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"(\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
        r"How much did\s*(\w+)\s*(\w+) at on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"Find the (\w+) price for\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"Show me the (\w+) price of\s*(\w+)\s*on\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\??",
        r"What was the (\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??",
        r"(\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})",
        r"How much did\s*(\w+)\s*(\w+) at on\s*(\d{4}-\d{2}-\d{2})\??",
        r"Find the (\w+) price for\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??",
        r"Show me the (\w+) price of\s*(\w+)\s*on\s*(\d{4}-\d{2}-\d{2})\??"
    ]

    # New forecast query patterns
    forecast_patterns = [
        r"(\w+)\s+Stock\s+Price\s+Prediction\s+on\s+(\d{1,2}/\d{1,2}/\d{4})\??",
        r"What\s+is\s+the\s+predicted\s+price\s+of\s+(\w+)\s+on\s+(\d{1,2}/\d{1,2}/\d{4})\??",
        r"Forecast\s+for\s+(\w+)\s+on\s+(\d{1,2}/\d{1,2}/\d{4})\??",
        r"(\w+)\s+price\s+forecast\s+for\s+(\d{1,2}/\d{1,2}/\d{4})\??",
        r"(\w+)\s+Stock\s+Forecast\s+on\s+(\d{1,2}/\d{1,2}/\d{4})\??"
    ]

    # Iterate over price patterns
    for pattern in price_patterns:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            price_type_raw = match.group(1).lower()
            symbol = match.group(2).upper()
            date_str = match.group(3)
            # Map the raw price_type to the metadata field
            price_type = PRICE_TYPE_MAPPING.get(price_type_raw)
            if not price_type:
                logger.error(f"Unknown price type in query: {price_type_raw}")
                return None, None, None
            try:
                # Parse date in different formats
                try:
                    date_obj = datetime.strptime(date_str, '%B %d, %Y')  # e.g., September 19, 2019
                except ValueError:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')  # e.g., 2019-09-19
                date = date_obj.strftime('%Y-%m-%d')  # Changed to YYYY-MM-DD
                return price_type, symbol, date
            except ValueError:
                logger.error(f"Invalid date format in query: {date_str}")
                return None, None, None

    # Iterate over forecast patterns
    for pattern in forecast_patterns:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            symbol = match.group(1).upper()
            date_str = match.group(2)
            try:
                date_obj = parser.parse(date_str)
                date = date_obj.strftime('%Y-%m-%d')  # Change to YYYY-MM-DD
                return 'predicted_price', symbol, date
            except (ValueError, parser.ParserError):
                logger.error(f"Invalid date format in forecast query: {date_str}")
                return None, None, None

    return None, None, None

@lru_cache(maxsize=256)
def get_stock_price(price_type, symbol, date):
    """
    Retrieves the specified stock price from Pinecone based on the price type, symbol, and date.
    """
    # Define metadata filter
    metadata_filter = {
        "type": {"$eq": "stock"},
        "symbol": {"$eq": symbol},
        "date": {"$eq": date}
    }
    # Perform a metadata-filtered search with a generic vector
    generic_query = "stock price query"
    generic_embedding = embedding_model.encode(generic_query).tolist()
    try:
        search_response = index.query(
            vector=generic_embedding,
            top_k=1,
            include_values=False,
            include_metadata=True,
            filter=metadata_filter
        )
        logger.debug(f"Pinecone search response: {search_response}")
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return "An error occurred while retrieving stock data."

    # Check if a match was found
    matches = search_response.get('matches', [])
    if matches:
        metadata = matches[0]['metadata']
        price = metadata.get(price_type, 'Unknown')
        symbol = metadata.get('symbol', 'Unknown')
        date = metadata.get('date', 'Unknown')
        logger.debug(f"Retrieved metadata: {metadata}")
        return f"The {price_type} price of {symbol} on {date} was ${price}."
    else:
        logger.info(f"No stock data found for {symbol} on {date}.")
        return f"No stock data found for {symbol} on {date}."

@lru_cache(maxsize=256)
def get_forecast_price(symbol, date):
    """
    Retrieves the forecasted stock price from Pinecone based on the symbol and date.
    """
    metadata_filter = {
        "type": {"$eq": "forecast"},
        "symbol": {"$eq": symbol},
        "date": {"$eq": date}
    }
    specific_query = f"forecast price for {symbol} on {date}"
    specific_embedding = embedding_model.encode(specific_query).tolist()
    try:
        search_response = index.query(
            vector=specific_embedding,
            top_k=3,  # Increased to retrieve multiple potential matches
            include_values=False,
            include_metadata=True,
            filter=metadata_filter
        )
        logger.debug(f"Pinecone search response (forecast): {search_response}")
    except Exception as e:
        logger.error(f"Error querying Pinecone (forecast): {e}")
        return "An error occurred while retrieving forecast data."

    matches = search_response.get('matches', [])
    if matches:
        logger.info(f"Found {len(matches)} forecast matches for {symbol} on {date}.")
        # Assuming the most relevant match is the first one
        metadata = matches[0]['metadata']
        predicted_price = metadata.get('predicted_price', 'Unknown')
        symbol = metadata.get('symbol', 'Unknown')
        date = metadata.get('date', 'Unknown')
        logger.debug(f"Retrieved forecast metadata: {metadata}")
        return f"The predicted price of {symbol} on {date} is ${predicted_price}."
    else:
        logger.info(f"No forecast data found for {symbol} on {date}.")
        return f"No forecast data found for {symbol} on {date}."

def chatbot_response(query):
    """
    Generates a response to the user's query by determining its type and retrieving relevant information.
    """
    try:
        logger.info(f"Received query: {query}")

        # Check if the query is a simple greeting or a general casual phrase
        if query.lower() in ['hello', 'hi', 'hey', 'greetings', 'howdy', 'good morning', 'good afternoon', 'good evening']:
            logger.info(f"Detected greeting query: {query}")
            return "Hello! How can I assist you today with financial markets, stocks, or related news?"
        # Check if the query is a thank-you message
        if re.search(r'\b(thanks?|thank you|i appreciate|much appreciated|cảm ơn)\b', query.lower()):
            logger.info("Detected thank-you message.")
            return "You're welcome! Let me know if you have any other questions about stocks or related news."

        # Detect if the query is requesting tweets by a specific writer
        tweet_query_match = re.match(r"List of tweets by writer named\s*:\s*(.+)", query, re.IGNORECASE)

        if tweet_query_match:
            # Handle tweet queries
            writer_name = tweet_query_match.group(1).strip().lower()
            logger.info(f"Detected specific query for writer: {writer_name}")

            # Define metadata filter
            metadata_filter = {
                "type": {"$eq": "tweet"},
                "writer": {"$eq": writer_name}
            }

            # Generate a generic embedding vector
            generic_query = "retrieve all tweets"
            try:
                generic_embedding = embedding_model.encode(generic_query).tolist()
            except Exception as e:
                logger.error(f"Error generating embedding for generic query: {e}")
                return "An error occurred while processing your request."

            # Perform a metadata-filtered search with the generic vector
            try:
                search_response = index.query(
                    vector=generic_embedding,
                    top_k=100,
                    include_values=False,
                    include_metadata=True,
                    filter=metadata_filter
                )
                logger.debug(f"Pinecone search response for tweets: {search_response}")
            except Exception as e:
                logger.error(f"Error querying Pinecone for tweets: {e}")
                return "An error occurred while retrieving tweets."

            # Check if any matches were found
            matches = search_response.get('matches', [])
            if not matches:
                logger.info(f"No tweets found for author '{writer_name}'.")
                return f"No tweets found for author '{writer_name}'."

            # Collect tweets from the matches
            tweets = []
            for match in matches:
                metadata = match.get('metadata', {})
                text = metadata.get('text', 'No content')
                post_date = metadata.get('post_date', 'Unknown Date')
                tweets.append(f"- {text} (Posted on {post_date})")

            # Optionally limit the number of tweets displayed
            max_display = 10
            tweets = tweets[:max_display]

            # Construct the response
            tweets_list = "\n".join(tweets)
            response = f"Here are some tweets by {writer_name.capitalize()}:\n{tweets_list}"
            logger.info(f"Responding with tweets for writer '{writer_name}'.")
            return response

        else:
            # Check if the query is asking for a specific stock price type on a specific date
            price_type, symbol, date = parse_stock_query(query)
            if price_type and symbol and date:
                if price_type == 'predicted_price':
                    # Handle forecast queries
                    logger.info(f"Detected forecast price query for symbol: {symbol} on date: {date}")
                    response = get_forecast_price(symbol, date)
                    logger.info(f"Responding with forecast price: {response}")
                    return response
                else:
                    # Handle regular stock price queries
                    logger.info(f"Detected {price_type} price query for symbol: {symbol} on date: {date}")
                    response = get_stock_price(price_type, symbol, date)
                    logger.info(f"Responding with stock price: {response}")
                    return response

            # Handle general queries using RAG
            logger.info("Detected general query. Performing vector similarity search.")

            # Generate embedding for the user's query
            try:
                query_embedding = embedding_model.encode(query).tolist()
                logger.debug(f"Generated query embedding.")
            except Exception as e:
                logger.error(f"Error generating embedding for query: {e}")
                return "An error occurred while processing your query."

            # Perform a similarity search in Pinecone
            try:
                search_response = index.query(
                    vector=query_embedding,
                    top_k=3,  # Adjusted for better context
                    include_values=False,
                    include_metadata=True,
                    filter={
                        "type": {"$in": ["news", "news2", "stock", "tweet", "forecast"]}
                    }
                )
                logger.debug(f"Pinecone search response for general query: {search_response}")
            except Exception as e:
                logger.error(f"Error querying Pinecone for general query: {e}")
                return "An error occurred while retrieving information."

            # Check if any matches were found
            matches = search_response.get('matches', [])
            if not matches:
                logger.info("No matches found in Pinecone.")
                return "There is not enough information to answer confidently."

            # Log the matches for debugging
            logger.info(f"Number of matches found: {len(matches)}")
            for idx, match in enumerate(matches):
                logger.debug(f"Match {idx}: Metadata: {match['metadata']}")

            # Construct the prompt using retrieved information
            prompt = construct_prompt(query, matches)
            logger.debug(f"Constructed prompt for model:\n{prompt}")

            if not (tokenizer and model):
                logger.error("Tokenizer or model not initialized.")
                return "Sorry, the system is currently experiencing issues and cannot generate a response right now."

            # Tokenize the prompt
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True,
                    add_special_tokens=True  # Ensure special tokens are included
                ).to(model.device)
                logger.debug(f"Tokenized inputs prepared.")
            except Exception as e:
                logger.error(f"Error tokenizing prompt: {e}")
                return "An error occurred while processing your request."

            # Generate response from the model
            try:
                with torch.no_grad():
                    # Use sampling for more diverse responses
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=256,  # Increased to allow for longer responses
                        do_sample=True,
                        top_p=0.85,          # Adjusted for better diversity
                        temperature=0.5,    # Slightly lower for more focused responses
                        num_return_sequences=1,  # Return one response
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )
                logger.debug(f"Model outputs generated.")
            except Exception as e:
                logger.error(f"Error generating response from model: {e}")
                return "An error occurred while generating a response."

            # Extract the generated response
            try:
                generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                logger.debug(f"Decoded raw response: {response}")

                # Cắt nếu model sinh thêm "Question: ..." lặp
                response = response.split("Question:")[0].strip()

                # Loại bỏ các nhãn không mong muốn
                response = re.sub(r"(headline|context|source|summary|symbol|date|reason):?", "", response, flags=re.IGNORECASE)

                # Làm sạch khoảng trắng thừa
                response = re.sub(r"\s+", " ", response).strip()

                # Cắt thành câu
                

                sentences = re.split(r'(?<=[.!?])\s+', response)

                # Lọc bỏ câu quá ngắn hoặc bị cắt
                filtered = []
                for s in sentences:
                    s = s.strip()
                    if len(s.split()) >= 2 and not s.lower().startswith("as an ai"):
                        filtered.append(s)
                from difflib import SequenceMatcher

                def is_redundant(a, b):
                    return SequenceMatcher(None, a, b).ratio() > 0.9

                unique_sentences = []
                for s in filtered:
                    if not any(is_redundant(s, u) for u in unique_sentences):
                        unique_sentences.append(s)

                filtered = unique_sentences

                if not filtered:
                    return "I'm sorry, I couldn't generate a useful response based on the context."

                # Câu đầu tiên là kết luận (Yes/No)
                main_answer = filtered[0]
                if not filtered[0].lower().startswith(("yes", "no")):
                    logger.warning("⚠️ Model did not start with Yes or No as required.")

                # Các câu sau là lý do hợp lý
                reason_keywords = ['because', 'due to', 'as', 'since']
                reasons = [
                    s for s in filtered[1:]
                    if any(kw in s.lower() for kw in reason_keywords)
                ]

                # Ghép kết quả
                final_parts = [main_answer] + reasons[:3]
                response = " ".join(final_parts)

                # Đảm bảo kết thúc bằng dấu chấm
                if not response.endswith('.'):
                    response += '.'

                logger.info(f"Final cleaned response: {response}")
            except Exception as e:
                logger.error(f"Error decoding or cleaning model output: {e}")
                return "An error occurred while processing the model's response."



            # Remove '**' from the response
            response = response.replace('**', '')

            # Clean up the response while preserving line breaks and numbered lists
            response_lines = response.split('\n')
            cleaned_lines = []
            for line in response_lines:
                stripped_line = line.strip()
                if stripped_line:
                    cleaned_lines.append(stripped_line)
            response = '\n'.join(cleaned_lines)
            if not response.endswith('.'):
                response += '.'
            logger.info(f"Generated response: {response}")

            # Fallback mechanism
            if not response:
                return "I'm sorry, I couldn't generate a response based on the provided information."
            else:
                return response

    except Exception as e:
        logger.error(f"Error in chatbot_response: {e}")
        return "An unexpected error occurred while processing your request."


# ------------------------------
# 3. Flask API Setup
# ------------------------------

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    """
    API endpoint to handle chatbot queries.
    Expects JSON payload with a 'query' field.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Please provide a query in JSON format.'}), 400
    query = data['query']
    response = chatbot_response(query)
    return jsonify({'response': response})

# ------------------------------
# 4. Run the Flask App
# ------------------------------

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)