# PreBuyPC

## Overview

PreBuyPC is a machine learning-powered web app that helps users choose the right laptop before buying. Users enter their budget and use case. The system evaluates available laptops and returns ranked recommendations.

Results stay locked until the user completes a payment.

## Core Features

- Multi-step recommendation form
- Machine learning ranking using XGBoost
- Real laptop database
- Payment integration with M-Pesa and Stripe
- Explanation engine for each recommendation

## How It Works

1. User lands on the homepage
2. User fills a multi-step form
3. System generates recommendations
4. User sees blurred preview
5. User pays to unlock full results

## Tech Stack

- Backend: Django 5
- Database: PostgreSQL
- Machine Learning: XGBoost, scikit-learn
- Payments: M-Pesa Daraja API, Stripe

## Project Structure

- `apps.laptops`: stores laptop data
- `apps.recommender`: ML engine and user flow
- `apps.payments`: payment handling
- `apps.accounts`: optional authentication
- `apps.dashboard`: landing page and entry point

## Setup Instructions

1. Clone the repository
2. Create virtual environment
3. Install dependencies

   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables

   Create a `.env` file and add:

   ```
   SECRET_KEY=your_key
   DEBUG=True
   DB_NAME=prebuypc
   DB_USER=postgres
   DB_PASSWORD=yourpassword
   ```

5. Run migrations

   ```
   python manage.py migrate
   ```

6. Run server

   ```
   python manage.py runserver
   ```

## Payment Flow

- M-Pesa uses STK push
- Stripe handles card payments
- Results unlock after payment confirmation

## Machine Learning

Model trained using labeled laptop dataset. Uses structured features from user query and laptop specs. Outputs probability score for ranking.

## Future Improvements

- Add user accounts and history
- Add admin dashboard analytics
- Improve recommendation explanations
- Add more data sources for laptops

**License:** All Rights Reserved License

**Author:** Maksim Felix