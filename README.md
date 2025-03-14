# Financial News Sentiment Analysis and Ranking

## Overview
This project is a multi-task model for financial news analysis. It performs two main tasks:
1. **Sentiment Analysis**: Classifies news articles as either `Positive` or `Negative`.
2. **Financial Importance Ranking**: Assigns a ranking score between `-10` and `10` to each article based on its financial significance.

The project uses a **BERT-based multi-task learning model** to achieve these tasks. It is designed to help financial analysts and investors quickly assess the sentiment and importance of news articles.

---

## Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/assermosa/financial-news-analysis.git
   cd financial-news-analysis