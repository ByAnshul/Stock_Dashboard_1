import sqlite3

def initialize_db():
    # Connect to SQLite database (or any other database you're using)
    conn = sqlite3.connect('stock_dashboard.db')
    cursor = conn.cursor()  # Initialize cursor here
    
    # Enable foreign key constraints
    cursor.execute('PRAGMA foreign_keys = ON')
    
    # Create tables for users, transactions, and holdings
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        balance REAL
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        transaction_type TEXT,
        ticker TEXT,
        price REAL,
        quantity INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS holdings (
        user_id INTEGER,
        ticker TEXT,
        quantity INTEGER,
        PRIMARY KEY (user_id, ticker),
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_summary (
        user_id INTEGER PRIMARY KEY,
        total_balance REAL DEFAULT 0,
        portfolio_value REAL DEFAULT 0,
        total_profit_loss REAL DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )''')
    
    conn.commit()
    conn.close()

# Call initialize_db function when this file is executed
if __name__ == "__main__":
    initialize_db()
    print("Database initialized successfully.")
