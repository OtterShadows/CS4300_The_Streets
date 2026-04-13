#!/usr/bin/env python
"""Script to regenerate character_data.pkl from the latest character_class code."""

import os
import sys
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from language_processing import character_class

# Load data
print("Loading comment and posting data...")
comments_df, postings_df = character_class.load_data()

# Create all characters
print("Creating character objects...")
characters = character_class.create_all_characters(postings_df, comments_df)

# Convert to dict
print("Converting characters to dictionary format...")
char_dict = character_class.characters_to_dict(characters)

# Save to pickle file
output_path = os.path.join(
    os.path.dirname(__file__), 
    'src', 
    'language_processing', 
    'data', 
    'character_data.pkl'
)

print(f"Saving character data to {output_path}...")
joblib.dump(char_dict, output_path)

print("✅ Successfully regenerated character_data.pkl!")
print(f"Total characters: {len(char_dict)}")
