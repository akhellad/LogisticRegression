# Logistic Regression Sentiment Analysis

## Overview
This project implements a logistic regression algorithm to predict the sentiment of user reviews, classifying them as either positive (1) or negative (0). Utilizing a dataset from Kaggle, the data is stored in a PostgreSQL database where tables are created and manipulated for analysis. This project involves data cleaning, splitting the dataset into training and testing sets, vectorization of the input for model compatibility, model creation and training, and finally, predicting the sentiment of new inputs.

## Getting Started

### Prerequisites
- Python 3.x
- PostgreSQL
- Libraries: pandas, sklearn, sqlalchemy, nltk

### Installation
1. Clone the repository to your local machine.
2. Install the required Python packages:
   ```bash
   pip install pandas sklearn sqlalchemy nltk
   ```
## Database Setup

1. **Setup your PostgreSQL database** and take note of your connection URL.

### Importing the Dataset

- Import the Kaggle dataset into your PostgreSQL database.

### Creating Tables

- Use PostgreSQL to create the necessary tables for managing the dataset.

### Configuring the Database Connection

- In the script, locate the `create_engine` line and replace the existing path with the URL to your database:
  ```python
  from sqlalchemy import create_engine
  engine = create_engine('YOUR_DATABASE_URL_HERE')
  ```
## Data Preparation

The script automatically processes the dataset by:

- Removing punctuation.
- Eliminating English stopwords such as "the", "not", etc.
- Splitting the data into training and testing sets.
- Vectorizing the input for model training and prediction.

## Training the Model

Run the script to train the logistic regression model on your dataset. The model will automatically handle data cleaning, training, and testing processes.

## Making Predictions

After training, the script will prompt for an input review. Enter the review, and the script will process and predict the sentiment (positive or negative) and output the corresponding score (0 or 1).

## Usage

To use the sentiment analysis model:

1. Ensure your PostgreSQL database is set up and accessible.
2. Run the script:
   ```bash
   python PredictReview.py
   ```
When prompted, enter a review to get the sentiment prediction.

## Contributing

Feel free to fork the repository and submit pull requests to contribute to the project.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Dataset provided by Kaggle.
- PostgreSQL for database management.
- Python libraries: `pandas` for data manipulation, `sklearn` for machine learning, `sqlalchemy` for database connection, and `nltk` for text processing.

