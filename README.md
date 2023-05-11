# README

## Installation

1. Git clone...
2. Create virtual env:

    ```bash/powershell/cmd
    python3 -m venv .venv
    ```

3. Activate virtual env:

    ```bash/zsh
    source .venv/bin/activate
    ```

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

    ```cmd
    .venv\Scripts\activate.bat
    ```

4. To contribute to the package, run from the project's root directory:

    ```bash/zsh
    pip install -e .
    ```

5. ...

## Supplying a New Labelled Dataset

You have two options:

1. Replace current labelled data:
   1. Replace the csv file in `./data/trainSet.csv` with your new dataset.
   2. **Important**: Make sure your file is also named `trainSet.csv`.
2. Modify `config.ini` file.
   1. Open the `./config.ini` file.
   2. Modify the file path in the line `path_labelled_data = ./data/trainSet.csv`.

Technically you can replace the `trainSet.csv` file with a file with a different name **AND** update the new file name in `config.ini`, but it's not necessary.

## Data Handling

### Split Labelled Dataset into Train, Validation, and Test Sets

To split the labelled data set into train, validation, and test. Default splitting is stratified per label category:

```bash
python3 main.py split
```

To split the data without stratifying it:

```bash
python3 main.py split --no-stratify
```

### Augment Train Set

To balance the train set classes, you can augment the train set by increasing its right skewness up to a specified percentile on the distribution of records per category. The default value for the percentile is 25, which means that all categories with less records than the records per category of the category at the 25th percentile will be upsampled (with replacement) up until they reach the same number of records as the category at the 25th percentile.

Instead of balancing the records per category up to a uniform distribution, this augmentation enables strengthening the signal from under-represented classes without significantly dampening the signal from the most representative classes in the data.

To augment the train data with the *default degree* of increased right skewness run:

```bash
python3 main.py augment
```

To augment the train data with with a *different degree* of increased right skewness run, e.g. up to the 50th percentile:

```bash
python3 main.py augment --percentile 50
```

## Transform Data to DatasetDict Format

We're using the `DatasetDict` class from the `datasets` library to format our data, since it stores data on disk in Arrow format, enabling efficient IO operations. Transformed data is localised at `./data/arrow_cache`.

For future operations, a proper external NoSQL database can be used to store and apply version control to our corpora, but we're storing datasets locally given the scope of the current task.

To transform the split and/or augmented data to a `DatasetDict` class and store on disk in Arrow format:

```bash
python3 main.py transform
```

## Training

To train models ... #TODO

```bash
python3 main.py train
```

## Evaluating Models

### Transformer Model

To evaluate distilbert model performance:

```bash
python3 main.py evaluate --model distilbert
```

Current model evaluates with the following metrics:

```bash
Evaluation results:
Accuracy = 0.66
Macro F1 Score = 0.60
Weighted F1 Score = 0.64
Macro Precision = 0.61
Weighted Precision = 0.65
Macro Recall = 0.62
Weighted Recall = 0.66
```

## Classifying

```bash
python3 main.py predict
```

## Assumptions

* A query can only be classified to a single category.

## Code Smells

* Trainer is reaching into DataHandler.

## Improvements

* Stream data instead of holding it in memory, so we can scale training and prediction to millions of search queries.
  * HuggingFace's `DatasetDict` class works with `Arrow` format and use data streaming.
  * But upstream we're reading and writing whole csv files; this can be improved.
* Use databases for data IO.
  * Could use a simple SQLite that comes pre-installed in Linux machines and python has `sqlite3` package out of the box.
  * My preference would be a NoSQL database, taking into account this application's future features.
