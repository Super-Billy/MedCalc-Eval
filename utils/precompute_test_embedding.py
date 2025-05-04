import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import json
from dotenv import load_dotenv
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up OpenAI client
load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Output file path
OUTPUT_PATH = "dataset_cleaned/embedding_key_data_new.jsonl"

# Load the dataset
async def load_data():
    return pd.read_csv("dataset_cleaned/train_data_no_leakage.csv")

# Prepare text to embed: Patient Note + Question
def build_input(row):
    return f"Note: {row['Patient Note'].strip()}\nQuestion: {row['Question'].strip()}"

# Create a semaphore to limit concurrent API calls
MAX_CONCURRENT_REQUESTS = 5
file_lock = asyncio.Lock()  # Lock for safe file access

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embedding(text, semaphore):
    async with semaphore:
        try:
            embedding_response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return embedding_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

async def process_row(row, semaphore):
    try:
        input_text = build_input(row)
        embedding = await get_embedding(input_text, semaphore)
        
        record = {
            "row_number": row["Row Number"],
            "embedding": embedding,
            "value": {
                "patient_note": row["Patient Note"],
                "question": row["Question"],
                "relevant_entities": row["Relevant Entities"],
                "ground_truth_answer": row["Ground Truth Answer"],
                "ground_truth_explanation": row["Ground Truth Explanation"]
            }
        }
        
        # Save this record immediately
        async with file_lock:
            async with aiofiles.open(OUTPUT_PATH, "a") as f:
                await f.write(json.dumps(record) + "\n")
                await f.flush()  # Ensure it's written to disk
                
        return record
    except Exception as e:
        logger.error(f"⚠️ Skipping row {row['Row Number']} due to error: {e}")
        return None

async def main():
    # Load data
    df = await load_data()
    df["input_text"] = df.apply(build_input, axis=1)
    
    # Initialize output file
    if os.path.exists(OUTPUT_PATH):
        # Get already processed rows
        processed_rows = set()
        try:
            async with aiofiles.open(OUTPUT_PATH, "r") as f:
                async for line in f:
                    data = json.loads(line)
                    processed_rows.add(data["row_number"])
            logger.info(f"Found {len(processed_rows)} already processed rows")
            
            # Filter rows to process
            df = df[~df["Row Number"].isin(processed_rows)]
            logger.info(f"Will process {len(df)} remaining rows")
        except Exception as e:
            logger.warning(f"Error reading existing file: {e}")
    else:
        # Create empty file
        open(OUTPUT_PATH, 'w').close()
    
    if len(df) == 0:
        logger.info("All rows have been processed already.")
        return
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Process in parallel with progress bar
    tasks = [process_row(row, semaphore) for _, row in df.iterrows()]
    records = await tqdm_asyncio.gather(*tasks, desc="Computing embeddings")
    
    # Count successful records
    valid_records = [r for r in records if r is not None]
    
    logger.info(f"✅ Processed {len(valid_records)} records in this run")
    logger.info(f"⚠️ Skipped {len(records) - len(valid_records)} records due to errors")

if __name__ == "__main__":
    asyncio.run(main())