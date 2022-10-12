### Instructions to run locally
1. Create a virtual env, activate it and install requirements
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2. To run in the Flask server
```
export FLASK_APP=app.py
flask run
```

### Notes
This project implements two endpoints namely
- `POST /analyze` with the body having a JSON that was 
   provided in the dataset. This endpoint returns a `task-id`.
   A sample JSON is available under `tests/this-american-life_getting-out.json`
   [Example postman query image](post_analyze.png)
- `GET /highlghts/<task-id>` returns the text highlight in this 
   podcast text. [Example postman query image](get_highlights.png)
- The topmost level script is `app.py`

### Algorithmic details
- The project aims to analyze the input transcription to extract
  sections of the podcast with topic changes. For this an algorith
  presented in the paper [Unsupervised Topic Segmentation of Meetings with BERT embeddings](https://arxiv.org/abs/2106.12978) 
  from Facebook. This method is heavily inspired by TextTiling. The for the paper is available [here](https://github.com/gdamaskinos/unsupervised_topic_segmentation)
- Modifications were made to this code to :
  - Use faster sentence embeddings from 
    the `sentence-transformer` package - [Link](https://github.com/UKPLab/sentence-transformers)
  - Incorporate bug-fixes to the original code 
  - Make the output usable
- Once a candidate segment is picked TextBlob was used to analyze
  the segment for sentiment subjectivity. The rationale here was that
  the more interesting the text the more opinionated and less subjective it might be.
  This was a quick-and-dirty solution to rank candidate topic segments.
  A more refined solution could have been presented if there were more time.

### Additional notes
- Additional notes made before starting the project are in [Notes.pdf](Notes.pdf)
- Comparing implementation outputs between my modifications and the Facebook paper -
  [Google Sheets Link](https://docs.google.com/spreadsheets/d/1NRnZKf4kS7ikICKWRuG4c2gtKvVkG59e-Vwb6JE28rs/edit?usp=sharing)