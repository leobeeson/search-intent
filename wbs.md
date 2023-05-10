# Work Breakdown Structure

## Run-Time Classification

### High-Level Design: Classification

* Classification Pipeline
  * Symbolic classification
    1. Exact Raw String Matcher #DONE
    2. Exact Preprocessed String Matcher
        * `Preprocessing Pipeline`
          * Removing whitespace and break/new lines
          * Lowercasing
          * Remove Punctuation
          * SKU Normaliser
    3. `Preprocessing Pipeline`
        * Unigram Tokenisation
        * MWE Tokenisation
    4. Sequential Keyword/MWE Matcher
    5. Bag-of-Keyword/MWE Matcher
    6. `Preprocessing Pipeline`
        * WordPiece Tokenisation
    7. WordPiece Root Matcher
  * Similarity Classification
    * Sentence Embeddings Matcher
    * Word Embeddings Matcher
* Ranker
* Scorer
* Test Set Analyser

## Modelling

### High-Level Design: Modelling

* Trainer
* Predictor
* Discrete Space Modelling
  * Keyword Extraction
  * MWE Extraction
  * Product/Service Extraction
  * Keyword/MWE-Category Keyness
* Continuous Space Modelling
  * Category Sentence Embeddings
  * Keyword/MWE Word Embedding

## Data Schemas

### Classification Pipeline

#### Exact Raw String Matcher

```json
{
  [
    "query": "int" // if matched 1 else 0 
  ]
}
```

#### Exact Preprocessed String Matcher

```json
{
  [
    "query_match": "category:int" // if matched 1 else 0 
  ]
}
```

#### Sequential Keyword/MWE Matcher

```json
{
  [
    ["query_match", "category:int", "score:float"]
  ]
}
```

#### Bag of Keywords Matcher

```json
{
  [
    ["query_match", "category:int", "score:float"]
  ]
}
```
