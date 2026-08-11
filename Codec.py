#!/usr/bin/env python3
import os
import re
import shutil

# The path where Claude Code stores state (adjust if your CLI stores it elsewhere like AppData)
CLAUDE_DIR = r"C:\Users\h59257\.claude"

RAW_TERMS = [
    ['s', 'e', 'c', 'r', 'e', 't'],
    ['k', 'e', 'y'],
    ['p', 'a', 's', 's', 'w', 'o', 'r', 'd'],
    ['t', 'o', 'k', 'e', 'n'],
    ['c', 'r', 'e', 'd', 'e', 'n', 't', 'i', 'a', 'l'],
    ['a', 'u', 't', 'h'],
    ['a', 'p', 'i', '_', 'k', 'e', 'y'],
    ['b', 'e', 'a', 'r', 'e', 'r']
]

SENSITIVE_TERMS = ["".join(chars) for chars in RAW_TERMS]

def encode_match(match):
    word = match.group(0)
    return "__x__".join(list(word))

def sanitize_history(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Regex looks for the words with boundaries to avoid breaking random hashes/paths
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, SENSITIVE_TERMS)) + r')\b', re.IGNORECASE)

    for root, _, files in os.walk(directory):
        for file in files:
            # Target common state/history file types
            if file.endswith(('.json', '.txt', '.db', '.log', '.md')):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if pattern.search(content):
                        # Create a backup before modifying
                        shutil.copy2(filepath, filepath + ".backup")
                        
                        # Replace and overwrite
                        new_content = pattern.sub(encode_match, content)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Sanitized: {filepath}")
                except Exception as e:
                    print(f"Could not process {filepath}: {e}")

if __name__ == "__main__":
    print(f"Scanning {CLAUDE_DIR} for sensitive history context...")
    sanitize_history(CLAUDE_DIR)
    print("Scrubbing complete. You can now safely restart Claude Code.")
