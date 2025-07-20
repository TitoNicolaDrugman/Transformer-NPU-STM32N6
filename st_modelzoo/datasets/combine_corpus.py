# FILE: combine_corpus.py
#
# A simple script to combine multiple .txt files from a 'datasets'
# folder into a single large corpus file. It also automatically removes
# the standard Project Gutenberg headers and footers.
#
# HOW TO USE:
# 1. Place this script inside the 'datasets' folder with your .txt files.
# 2. Open a terminal, 'cd' into the 'datasets' folder.
# 3. Run: python combine_corpus.py
#
import os
import re

# --- Configuration (Modified) ---
# This script is now designed to be run from *inside* the 'datasets' folder.
# Therefore, the source directory is the current directory ('.').
SOURCE_DIR = '.'

# Name of the final combined output file
OUTPUT_FILE = 'full_corpus.txt'

# List of the files you want to combine (from your screenshot)
# We explicitly list them to avoid including 'tinyshakespeare.txt' since we have the full version.
FILES_TO_COMBINE = [
    'shakespeare.txt',
    'moby_dick.txt',
    'pride_and_prejudice.txt',
    'frankenstein.txt',
]

# --- Main Script ---

def clean_and_read_file(filepath):
    """
    Reads a file and attempts to strip Project Gutenberg headers/footers.
    """
    print(f"  -> Processing: {os.path.basename(filepath)}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Regex to find the main content between the start and end markers
        # This is flexible to handle slight variations in the Gutenberg text
        start_match = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*', text)
        end_match = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*', text)

        if start_match and end_match:
            # If markers are found, slice the text between them
            start_pos = start_match.end()
            end_pos = end_match.start()
            print("      - Gutenberg markers found. Cleaning file.")
            return text[start_pos:end_pos].strip()
        else:
            # If no markers, assume it's a clean file and return the whole content
            print("      - No Gutenberg markers found. Using entire file.")
            return text.strip()

    except FileNotFoundError:
        print(f"  [ERROR] File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"  [ERROR] Could not process {filepath}: {e}")
        return ""

def main():
    """
    Main function to combine the specified text files.
    """
    print("--- Starting Corpus Combination ---")
    all_content = []

    for filename in FILES_TO_COMBINE:
        filepath = os.path.join(SOURCE_DIR, filename)
        content = clean_and_read_file(filepath)
        if content:
            all_content.append(content)

    if not all_content:
        print("\n[FATAL] No content was gathered. Please check file paths and names.")
        return

    # Create the output file in the same directory
    output_path = os.path.join(SOURCE_DIR, OUTPUT_FILE)
    print(f"\nWriting combined content to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Join all books with two newlines to act as a clear separator
        outfile.write('\n\n'.join(all_content))

    output_size = os.path.getsize(output_path) / (1024 * 1024) # in MB
    print("\n--- Success! ---")
    print(f"Combined {len(all_content)} files into '{output_path}'")
    print(f"Final corpus size: {output_size:.2f} MB")
    print("----------------")
    print("\nNext Step: Remember to update 'user_config.txt' to point to this new file.")


if __name__ == "__main__":
    main()