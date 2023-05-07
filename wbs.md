# Work Breakdown Structure

## Run-Time Classification

### High-Level Design: Classification

* Classification Pipeline
  * Symbolic classification
    1. Exact Raw String Matcher
    2. `Preprocessing Pipeline`
        * Lowercasing
        * Remove Punctuation
        * SKU Normaliser
    3. Exact Preprocessed String Matcher
    4. `Preprocessing Pipeline`
        * Unigram Tokenisation
        * MWE Tokenisation
    5. Sequential Keyword/MWE Matcher
    6. Bag-of-Keyword/MWE Matcher
    7. `Preprocessing Pipeline`
        * WordPiece Tokenisation
    8. WordPiece Root Matcher
  * Similarity Classification
    * Sentence Embeddings Matcher
    * Word Embeddings Matcher
* Ranker
* Validation Set Evaluator
* Test Set Analyser

## Modelling

### High-Level Design: Modelling

* Discrete Space Modelling
  * Keyword Extraction
  * MWE Extraction
  * Product/Service Extraction
  * Keyword/MWE-Category Keyness
* Continuous Space Modelling
  * Category Sentence Embeddings
  * Keyword/MWE Word Embedding
