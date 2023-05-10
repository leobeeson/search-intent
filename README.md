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



```bash
python3 main.py augment
```

## Training

```bash
python3 main.py train
```

## Validate Classifications

```bash
python3 main.py validate
```

## Classifying

```bash
python3 main.py predict
```

## Assumptions

* A query can only be classified to a single category.