[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Feature Extraction Optimization for SVM 

Provides examples of potential preprocessing techniques to improve SVM performance.
This repo is setup and tested to train on Google Cloud.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You'll need Python 3.5+.
You'll also need the Sentiment140 Dataset available at: http://help.sentiment140.com/for-students

Save this dataset to svm_trainer/data/ as "stanford140.csv"

Setup project as follows:
```
#setup virtual environment
virtualenv venv
source venv/bin/activate

# install python requirements
pip install -r requirements.txt

# run locally with:
python test.py
```
## Deployment

Google Cloud deployment instructions coming soon.

## Authors

* **Ben Krig** - *Initial work* - [Ben Krig](https://github.com/benkrig)
* **Salvatore Nicosia** - *Initial work*
* **Nick Schiffer** - *Initial work*
* **Darren Truong** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
