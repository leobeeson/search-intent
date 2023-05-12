# Work Breakdown Structure

## Run-Time Classification

### High-Level Design: Classification

* Classification Pipeline
  * Symbolic classification
    1. Exact Raw String Matcher                         #SHOULD #DONE
       * Acts as a durable cache.
    2. Exact Preprocessed String Matcher                #WONT #SOMEDAY
        * OUT-OF-SCOPE: *test data has already been preprocessed*
        * `Pipeline Transformers`
          * Removing whitespace and break/new lines
          * Lowercasing
          * Remove Punctuation
          * SKU Normaliser
    3. Sequential Keyword/MWE Matcher                    #COULD #SOON
        * `Pipeline Transformers`
          * Unigram Tokenisation
          * MWE Tokenisation
    4. Bag-of-Keyword/MWE Matcher                        #COULD #SOON
    5. WordPiece Root Matcher                            #SHOULD #NEXT
        * `Pipeline Transformers`
          * WordPiece Tokenisation
  * Similarity Classification
    1. Sequence Classification Transformer               #MUST #DONE
    2. Sentence Embeddings Matcher                       #SHOULD #NEXT
       * Use: `flax-sentence-embeddings/all_datasets_v4_MiniLM-L6`
    3. Word Embeddings Matcher                           #WONT #SOMEDAY
* Evaluator                                              #MUST #DONE
* Ranker                                                 #SHOULD #NEXT
* Test Set Analyser                                      #COULD #SOON

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
  * Sequence Classification Transformer                  #MUST #DONE
  * Category Sentence Embeddings
  * Keyword/MWE Word Embedding

### High-Level Design: Exploration

* Model Building:
  * Category Reification -> GPT #WIP
    * `global_category_label`: str
    * `global_topic_names`: list[str]
    * `global_nouns`: list[str]
    * `subcategory_category_label`: list[str]
    * `topics_category_label`: list[str]
    * `nouns_category_label`: list[str]
  * Category Expansion (UK & USA) -> SERP + GPT
    * `top_brand_names`: list[str]
    * `top_product_names`: list[str]
    * `typical_SKU_formats`: list[str]
  * Category Summarisation -> GPT
    * Define compact `category_summary` object for classification stage.
  * Category Indexing
    * Localised storage of compact data object.
* Classification:
  * Single query:                                        #COULD #SOON
    * ...
  * Batch:                                               #MUST #NOW
    * Search Query Batching (Verify number of tokens)
    * Search Queries + `category_summary` injection
    * Chat completion call
    * Response deserialization to json.
    * Evaluate results (against test data)
    * Store predictions.
* Training Data Curation:
  * Identify semantic outlier search queries per category.
  * Identify alternative likelier search query category.
  * Reclassify search queries categories.
* Transformer Model:
  * Transformer Model Retraining

#### Category Reification

* Extraction
  * ResponseSchema
  * StructuredOutputParser
  * ChatPromptTemplate
    * HumanMessagePromptTemplate

#### Training Data Curation

* Evaluation
  * QEvalChain

## TODO

* Load model to S3 #DONE
* requirements.txt #DONE
* Refactor and add scripts for transformer training. #DONE
* Refactor and add scripts for transformer hyperparameter optimization. #DONE
* Document hyperparameter optimization results. #WIP
* Finish and insert logical design. #DONE

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
