# AI Data Analyst

This project is an AI-powered data analyst application that allows you to chat with your data. It consists of a Django backend and a React frontend.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd ai_data_analyst
```

### 2. Backend Setup

Navigate to the backend directory and set up the Python environment.

```bash
cd backend
```

**Create a virtual environment:**

```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

**Activate the virtual environment:**

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Set up environment variables:**

Create a `.env` file in the `backend` directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Apply database migrations:**

```bash
python manage.py migrate
```

**Seed the database with demo data (optional, for testing):**

```bash
python seed.py
```

This will create a `sales.db` SQLite database with sample sales data including products, customers, orders, and order items. The data includes realistic patterns for testing RFM analysis, trend analysis, product performance, customer clustering, and more.

**Run the development server:**

```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`.

### 3. Frontend Setup

Open a new terminal window, navigate to the frontend directory, and install dependencies.

```bash
cd frontend
```

**Install dependencies:**

```bash
npm install
```

**Run the development server:**

```bash
npm run dev
```

The frontend application will be available at `http://localhost:5173` (or the port shown in your terminal).

## Usage

1. Ensure both backend and frontend servers are running.
2. Open your browser and navigate to the frontend URL.
3. Start chatting with your data!
