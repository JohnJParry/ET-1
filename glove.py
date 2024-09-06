import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools
import csv
import os
import json


def load_glove_embeddings(file_path:str) -> dict:
    """
    Load GloVe embeddings.
    Pre-trained weights download link can be found here: https://nlp.stanford.edu/projects/glove/
    Args:
        file_path: path to .txt file containing pre-trained GloVe weights. 
    Return:
        (dict[str, Tensor]) A dictionary containing each word and the associated vector.  

    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        for line in tqdm(f, total=total_lines, desc="Loading GloVe embeddings"):
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = torch.tensor(vector)
    return embeddings

def get_word_vector(word:str, embeddings:dict):
    """
    Returns the associated word vector. 
    Args:
        word (str): input word to embed. 
        embeddings (np.ndarray): pre-trained embedding array. 
    Return:
        numpy array containing associated word-vector. 

    """
    return embeddings.get(word.lower())

def cosine_similarity(vec1:Tensor, vec2:Tensor)->float:
    "Compute the cosine similarity between two vectors."
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cos_sim.item()

def compute_avg_cosine_distance_score(words:list, embeddings:dict[str, Tensor]):
    """
    Calculate Divergent Association Task (DAT) score. See https://www.pnas.org/doi/epub/10.1073/pnas.2022340118.
    We calculate the average score based on 7 extracted words. 
    Args: 
        words (list): list containing word strings to be evaluated. 
        embeddings (np.ndarray): pre-trained embedding array.
    Return:
        (float) DAT score. 
    """
    valid_words = [word for word in words if get_word_vector(word, embeddings) is not None][:7]
    
    if len(valid_words) < 2:
        return None 
    
    distances = []
    for word1, word2 in itertools.combinations(valid_words, 2):
        vec1 = get_word_vector(word1, embeddings)
        vec2 = get_word_vector(word2, embeddings)
        if vec1 is not None and vec2 is not None:
            cosine_dist = 1 - cosine_similarity(vec1, vec2)
            distances.append(cosine_dist)
    
    average_distance = sum(distances) / len(distances)
    score = average_distance * 100
    
    return score

def compute_similarity_matrix(words:list, embeddings:dict[str, Tensor]):
    """
    Compute the similarity matrix for a list of words. 
    Args: 
        words (list): list containing word strings to be evaluated. 
        embeddings (np.ndarray): pre-trained embedding array.
    Return:
        (np.ndarray) 7x7 symmetric matrix containing semantic distance scores.  
    """
    matrix = []
    for word1, word2 in itertools.product(words, repeat=2):
        pair_words = [word1, word2]
        score = compute_avg_cosine_distance_score(pair_words, embeddings)
        matrix.append((word1, word2, score))
    return matrix

def save_similarity_matrix_csv(matrix:np.ndarray, words:list, output_file:str):
    """
    Save similarity matrix into .csv file in matrix format for displaying in sheets. 
    Note that scores are rounded to be consistent with scores presented in 
    https://www.pnas.org/doi/epub/10.1073/pnas.2022340118. 
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = [''] + words
        csvwriter.writerow(header)
        for i, word1 in enumerate(words):
            row = [word1]
            for j, word2 in enumerate(words):
                score = matrix[i * len(words) + j][2]
                if score is None:
                    row.append('NOT FOUND')
                else:
                    row.append(round(score))
            csvwriter.writerow(row)

def save_summary_csv(summary_data:list, output_file:str):
    """
    Save average scores for each experiment to .csv file. 
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File Name', 'Overall DAT Score'])
        for row in summary_data:
            csvwriter.writerow(row)

def process_word_lists_from_json(json_file:str, embeddings:dict[str, Tensor], output_dir:str):
    """
    Batch process LLM generated list of words contained in .json file. 
    .json file should be formatted with enteries:

        "experiment name": {
                "filename": "path/to/file.csv"m
                "words": ["list", "of", ..., "words"]
                }
    """
    with open(json_file, 'r') as f:
        word_lists = json.load(f)

    summary_data = []

    for list_name, data in word_lists.items():
        words = data['words']
        output_file_path = os.path.join(output_dir, data['filename'])
        
        print(f"Processing word list '{list_name}': {words}")
        
        overall_dat_score = compute_avg_cosine_distance_score(words, embeddings)
        if overall_dat_score is None:
            print(f"Not enough valid words to compute the DAT score for '{list_name}'")
            continue
        
        summary_data.append((data['filename'], round(overall_dat_score, 2)))

        valid_words = [word for word in words if get_word_vector(word, embeddings) is not None][:7]
        similarity_matrix = compute_similarity_matrix(valid_words, embeddings)
        save_similarity_matrix_csv(similarity_matrix, valid_words, output_file_path)
        
        print(f"Similarity matrix saved to {output_file_path}")

    summary_file_path = os.path.join(output_dir, 'summary_dat_scores.csv')
    save_summary_csv(summary_data, summary_file_path)
    print(f"Summary of DAT scores saved to {summary_file_path}")

def main():
    glove_file_path = '/home/balthier/Downloads/glove.42B.300d.txt'
    glove_embeddings = load_glove_embeddings(glove_file_path)
    json_file_path = './word_lists.json'
    output_directory = './output'
    os.makedirs(output_directory, exist_ok=True)
    process_word_lists_from_json(json_file_path, glove_embeddings, output_directory)

if __name__ == "__main__":
    main()

