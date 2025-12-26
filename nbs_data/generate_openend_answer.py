from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Global client - will be initialized once per worker process
_client = None


def init_worker(api_key, base_url):
    """Initialize the OpenAI client once per worker process."""
    global _client
    _client = OpenAI(api_key=api_key, base_url=base_url)


def answer_expand(example_answer, n=10, max_retries=3):
    """Generate diverse answers using LLM API."""
    global _client
    
    prompt = f"""
    Generate {n} diverse answers to a general question based on the example answers provided.
    Ensure the answers are varied in phrasing but consistent in meaning. You can switch between short and long answers, and use capitalization or different wording.
    You should preserve all key information (such as sex, age group, cognition, disease, etc.) from the example answers and not introduce any new facts.
    Format the output as a list, with each answer on a new line. Do not include any numbering or bullet points.

    Example Answers:
    {', '.join(example_answer)}
    """
    
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            
            text = response.choices[0].message.content.strip()
            
            # Split lines into answers (depending on how the LLM formats output)
            answers = [line.strip() for line in text.split("\n") if line.strip()]
            
            return answers
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                print(f"API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Last attempt failed, raise a simple error
                raise RuntimeError(f"API Error after {max_retries} attempts: {type(e).__name__}: {str(e)}")

def get_desc(x):
    if x > 1.5:
        return 'above average for'
    elif x < -1.5:
        return 'below average for'
    else:
        return 'average for'

def process_row(row):
    """Process a single row to generate answers."""
    age = row['age_desc']
    sex = 'male' if row['sex'] else 'female'
    
    # fluid_intel = get_desc(row['fluid_intelligence_z'])
    # fluid_intel += ' adults over 30'
    # base_answer = f"The subject is a {age} {sex} with a fluid intelligence score that is {fluid_intel}."

    # fluidcomp = get_desc(row['fluid_composite_z'])
    # fluidcomp += ' adults over 30'
    # flanker = get_desc(row['flanker_score_z'])
    # flanker += ' adults over 30'
    # base_answer = f"The subject is a {age} {sex} with a fluid composite score that is {fluidcomp} and a flanker score that is {flanker}."

    AD_diag = {0: 'cognitively normal', 1: 'mild cognitive impairment', 2: 'Alzheimer\'s disease'}
    diag = AD_diag.get(row['diagnosis'], 'of unknown diagnosis')
    apoe4 = {0: 'no APOE4 alleles', 1: 'one APOE4 allele', 2: 'two APOE4 alleles'}
    apoe4_status = apoe4.get(row['apoe4'], 'unknown APOE4 status')
    av45 = {0: 'AV45 negative', 1: 'AV45 positive'}
    av45_status = av45.get(row['av45'], 'unknown AV45 status')
    base_answer = f"The subject is a {age} {sex} who is {diag} with {apoe4_status} and {av45_status}."

    # Generate answers using the LLM API
    answers = []
    for _ in range(2):
        expanded = answer_expand([base_answer], n=10)
        answers.extend(expanded)
    
    answers.append(base_answer)  # Ensure base answer is included
    answers = list(set(answers))  # Deduplicate

    return {
        'subject_id': row['subject_id'],
        'session_id': row['session_id'],
        'answers': answers
    }


def generate_openended_answers(input_csv, output_csv, num_processes=None, batch_size=100):
    """Generate open-ended answers using multiprocessing with batch saving.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        num_processes: Number of processes to use (default: cpu_count())
        batch_size: Number of rows to process before saving (default: 100)
    """
    df = pd.read_csv(input_csv)
    
    # Get API credentials
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = "https://api.deepseek.com"
    
    # Set number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), len(df))
    
    # Check for existing progress and resume
    temp_output = output_csv.replace('.csv', '_temp.csv')
    processed_ids = set()
    
    if os.path.exists(temp_output):
        print(f"Found existing progress file: {temp_output}")
        existing_df = pd.read_csv(temp_output)
        processed_ids = set(zip(existing_df['subject_id'], existing_df['session_id']))
        print(f"Resuming from {len(processed_ids)} already processed rows...")
    
    # Filter out already processed rows
    if processed_ids:
        df['_processed'] = df.apply(lambda row: (row['subject_id'], row['session_id']) in processed_ids, axis=1)
        remaining_df = df[~df['_processed']].drop(columns=['_processed'])
        df = df.drop(columns=['_processed'])
    else:
        remaining_df = df
    
    if len(remaining_df) == 0:
        print("All rows already processed! Moving temp file to final output...")
        if os.path.exists(temp_output):
            os.rename(temp_output, output_csv)
        print(f"Results saved to {output_csv}")
        return
    
    print(f"Processing {len(remaining_df)} remaining rows using {num_processes} processes...")
    print(f"Saving progress every {batch_size} rows...")
    
    # Convert DataFrame rows to list of dictionaries for multiprocessing
    rows = [row for _, row in remaining_df.iterrows()]
    
    # Process rows in batches with progress bar
    all_results = []
    
    try:
        # Initialize pool with worker initialization function
        with Pool(processes=num_processes, initializer=init_worker, initargs=(api_key, base_url)) as pool:
            pbar = tqdm(total=len(rows), desc="Generating answers")
            
            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                batch_results = []
                
                # Process batch
                for result in pool.imap(process_row, batch):
                    batch_results.append(result)
                    all_results.append(result)
                    pbar.update(1)
                
                # Save batch to temp file
                batch_df = pd.DataFrame(batch_results)
                
                if os.path.exists(temp_output):
                    # Append to existing temp file
                    existing_df = pd.read_csv(temp_output)
                    combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    combined_df.to_csv(temp_output, index=False)
                else:
                    # Create new temp file
                    batch_df.to_csv(temp_output, index=False)
                
                pbar.set_postfix({'saved': len(all_results), 'file': 'temp'})
            
            pbar.close()
        
        # Move temp file to final output
        print(f"\nProcessing complete! Moving temp file to final output...")
        os.rename(temp_output, output_csv)
        print(f"Results saved to {output_csv}")
        
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Progress saved to {temp_output}")
        print(f"Run the script again to resume from where you left off.")
        raise
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print(f"Progress saved to {temp_output}")
        print(f"Run the script again to resume from where you left off.")
        raise


if __name__ == "__main__":
    generate_openended_answers(
        'data/ADNI/fmri/metadata_with_text_medical_gpt.csv',
        'data/ADNI/fmri/answers_open.csv',
        num_processes=32,  # Adjust based on your system and API rate limits
        batch_size=100  # Save progress every 100 rows
    )