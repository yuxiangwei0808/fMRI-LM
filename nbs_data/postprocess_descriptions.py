import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Initialize OpenAI client
os.environ["DEEPSEEK_API_KEY"] = ""
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

prompt = """You are an expert neuroscientist tasked with rewriting fMRI brain {descriptor_type} descriptions to be more diverse and natural while maintaining scientific accuracy.

Your task:
- Rewrite the given description to sound more natural and varied
- Preserve all numerical z-score values exactly as provided
- Maintain the scientific interpretation and conclusions
- Vary sentence structure, vocabulary, and phrasing to avoid template-like language
- Keep the same level of technical detail
- Use synonyms and alternative phrasings (e.g., "connectivity" vs "functional coupling", "elevated" vs "increased" vs "enhanced", "network" vs "system")
- Reorganize information presentation order when appropriate
- Keep the rewritten description roughly the same length as the original

Important:
- DO NOT add new interpretations or conclusions not present in the original
- DO NOT remove any important quantitative information (z-scores, network names, metric values)
- DO NOT change the clinical/scientific meaning
- Maintain a professional, scientific tone

Output only the rewritten description without any preamble or explanation."""

DESC_TYPES = {'fc_descriptors.csv': 'functional connectivity-based',
              'gradient_descriptors.csv': 'functional gradient-based',
              'graph_descriptors.csv': 'graph metric-based',
              'ica_descriptors.csv': 'independent component analysis-based',
              'fc_self_descriptors.csv': 'functional connectivity-based',
              'region_self_descriptors.csv': 'average regional activity-based'}

def rewrite_description(desc, desc_type, model="deepseek-chat", max_retries=3, api_key=None, base_url=None):
    """
    Rewrite a single fMRI description using OpenAI API.
    
    Parameters:
    -----------
    desc : str
        Original description to rewrite
    desc_type : str
        Type of descriptor (for prompt customization)
    model : str
        OpenAI model to use (default: deepseek-chat)
    max_retries : int
        Maximum number of retry attempts on API failure
    api_key : str, optional
        API key (if not provided, uses environment variable)
    base_url : str, optional
        Base URL for API (if not provided, uses default)
    
    Returns:
    --------
    str
        Rewritten description, or original if API fails
    """
    # Create client (needed for multiprocessing as each process needs its own client)
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    if base_url is None:
        base_url = "https://api.deepseek.com"
    
    client_local = OpenAI(api_key=api_key, base_url=base_url)
    
    desc_prompt = prompt.format(descriptor_type=desc_type)
    for attempt in range(max_retries):
        try:
            response = client_local.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": desc_prompt},
                    {"role": "user", "content": f"Original description:\n\n{desc}"}
                ],
                temperature=0.8,  # Higher temperature for more diversity
                max_tokens=1000
            )
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Failed to rewrite after {max_retries} attempts: {str(e)[:100]}")
                raise Exception
    
    return desc

def rewrite_description_wrapper(args):
    """
    Wrapper function for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (index, description, desc_type, model, max_retries, api_key, base_url)
    
    Returns:
    --------
    tuple
        (index, rewritten_description or None if failed)
    """
    idx, desc, desc_type, model, max_retries, api_key, base_url = args
    try:
        rewritten = rewrite_description(desc, desc_type, model, max_retries, api_key, base_url)
        return (idx, rewritten)
    except Exception as e:
        print(f"  Warning: Failed to rewrite row {idx}: {str(e)[:100]}")
        return (idx, None)  # Return None to mark as failed, allowing auto-resume

def process_file(file, input_dir='data/UKB/fmri/descriptors', output_dir='data/UKB/fmri/descriptors_rewritten', 
                 batch_size=10, use_multiprocessing=True, max_workers=4, model="deepseek-chat"):
    """
    Process a single CSV file and rewrite all descriptions.
    
    Parameters:
    -----------
    file : str
        Filename of the CSV to process
    input_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory to save rewritten CSV files
    batch_size : int
        Number of rows to process before saving checkpoint
    use_multiprocessing : bool
        Whether to use multiprocessing for parallel processing
    max_workers : int
        Maximum number of parallel processes (default: 4)
        If None, uses cpu_count()
    model : str
        Model to use for rewriting
    """
    input_path = os.path.join(input_dir, file)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file)
    
    print(f"\nProcessing {file}...")
    desc_type = DESC_TYPES[file]
    
    # Determine number of workers
    if max_workers is None:
        max_workers = cpu_count()
    else:
        max_workers = min(max_workers, cpu_count())
    
    print(f"  Using {'multiprocessing' if use_multiprocessing else 'sequential'} mode" + 
          (f" with {max_workers} workers" if use_multiprocessing else ""))
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"  Output file already exists. Loading existing progress...")
        df = pd.read_csv(output_path)
        
        # Check if we need to continue processing
        if 'summary_rewritten' in df.columns:
            remaining = df['summary_rewritten'].isna().sum()
            if remaining == 0:
                print(f"  All descriptions already rewritten. Skipping.")
                return df
            else:
                print(f"  {remaining} descriptions remaining to rewrite.")
    else:
        df = pd.read_csv(input_path)
        df['summary_rewritten'] = None
    
    # Process rows that don't have rewritten summaries yet
    rows_to_process = df[df['summary_rewritten'].isna()].index.tolist()
    
    if not rows_to_process:
        print("  No rows to process.")
        return df
    
    # Get API credentials
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = "https://api.deepseek.com"
    
    if use_multiprocessing and len(rows_to_process) > 1:
        # Multiprocessing mode
        print(f"  Processing {len(rows_to_process)} descriptions in batches...")
        
        # Process in chunks for checkpointing
        for chunk_start in range(0, len(rows_to_process), batch_size):
            chunk_end = min(chunk_start + batch_size, len(rows_to_process))
            chunk_indices = rows_to_process[chunk_start:chunk_end]
            
            # Prepare arguments for multiprocessing
            args_list = [
                (idx, df.at[idx, 'summary'], desc_type, model, 3, api_key, base_url)
                for idx in chunk_indices
                if pd.notna(df.at[idx, 'summary'])
            ]
            
            # Process chunk with multiprocessing
            with Pool(processes=max_workers) as pool:
                results = list(tqdm(
                    pool.imap(rewrite_description_wrapper, args_list),
                    total=len(args_list),
                    desc=f"  Batch {chunk_start//batch_size + 1}/{(len(rows_to_process)-1)//batch_size + 1}"
                ))
            
            # Update dataframe with results
            failed_count = 0
            for idx, rewritten in results:
                if rewritten is not None:
                    df.at[idx, 'summary_rewritten'] = rewritten
                else:
                    # Keep as NaN to allow auto-resume
                    failed_count += 1
            
            if failed_count > 0:
                print(f"  Warning: {failed_count} descriptions failed in this batch and will be retried on next run")
            
            # Save checkpoint
            df.to_csv(output_path, index=False)
            
    else:
        # Sequential mode (original implementation)
        print(f"  Processing {len(rows_to_process)} descriptions sequentially...")
        
        for idx in tqdm(rows_to_process, desc=f"  Rewriting {file}"):
            if pd.notna(df.at[idx, 'summary']):
                original_desc = df.at[idx, 'summary']
                try:
                    rewritten_desc = rewrite_description(original_desc, desc_type=desc_type, model=model)
                    df.at[idx, 'summary_rewritten'] = rewritten_desc
                except Exception as e:
                    print(f"  Warning: Failed to rewrite row {idx}: {str(e)[:100]}")
                    # Leave as NaN to allow auto-resume
                    pass
                
                # Save checkpoint every batch_size rows
                if (idx + 1) % batch_size == 0:
                    df.to_csv(output_path, index=False)
            else:
                print(f"  Warning: Row {idx} has no summary to rewrite.")
    
    # Final save
    df.to_csv(output_path, index=False)
    print(f"  Completed! Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rewrite fMRI descriptions using LLM')
    parser.add_argument('--sequential', action='store_true', 
                       help='Use sequential processing instead of multiprocessing')
    parser.add_argument('--max-workers', type=int, default=64,
                       help='Maximum number of parallel workers (default: 64)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Number of descriptions to process before saving checkpoint (default: 64)')
    parser.add_argument('--files', nargs='+', 
                       choices=['fc_descriptors.csv', 'gradient_descriptors.csv', 
                               'graph_descriptors.csv', 'ica_descriptors.csv',
                               'fc_self_descriptors.csv', 'region_self_descriptors.csv'],
                       help='Specific files to process (default: all)')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                       help='Model to use (default: deepseek-chat)')
    parser.add_argument('--input-dir', type=str, default='data/UKB/fmri/descriptors',
                       help='Input directory containing descriptor CSV files')
    
    args = parser.parse_args()
    
    # List of descriptor files to process
    if args.files:
        files = args.files
    else:
        files = ['fc_descriptors.csv', 'gradient_descriptors.csv', 
                'graph_descriptors.csv', 'ica_descriptors.csv',
                'fc_self_descriptors.csv', 'region_self_descriptors.csv']
    
    use_multiprocessing = not args.sequential
    
    print("="*60)
    print("fMRI Description Rewriting")
    print("="*60)
    print(f"Mode: {'Multiprocessing' if use_multiprocessing else 'Sequential'}")
    if use_multiprocessing:
        print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Files to process: {files}")
    print("="*60)
    
    # Process each file
    results = {}
    output_dir = args.input_dir.replace('descriptors', 'descriptors_rewritten')
    os.makedirs(output_dir, exist_ok=True)
    for file in files:
        try:
            df = process_file(
                file, 
                input_dir=args.input_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                use_multiprocessing=use_multiprocessing,
                max_workers=args.max_workers,
                model=args.model
            )
            results[file] = df
        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All files processed successfully!")
    print("="*60)