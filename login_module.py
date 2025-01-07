import streamlit as st
import sqlite3
from hashlib import sha256

# Hash passwords
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Register user
def register_user(username, email, password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""SELECT username, email FROM users WHERE username = ? AND password = ?""", 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

# Login or register UI
def login_ui():
    st.title("User Authentication")
    tabs = st.tabs(["Login", "Register"])

    # Login tab
    with tabs[0]:
        st.subheader("Login")
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1,3])

        with col1:
        
            if st.button("Login"):
                user = authenticate_user(login_user, login_pass)
                if user:
                    st.success(f"Welcome {user[0]}!")  # Personalized welcome message
                    st.session_state.logged_in = True
                    st.session_state.username = user[0]
                    st.session_state.email = user[1] # Store username in session state
                else:
                    st.error("Invalid credentials.")
        with col2:
            if st.button("Skip"):
                st.session_state.logged_in = True


    # Register tab
    with tabs[1]:
        st.subheader("Register")
        reg_user = st.text_input("Choose a Username")
        reg_email = st.text_input("Enter your Email")
        reg_pass = st.text_input("Choose a Password", type="password")
        if st.button("Register"):
            if register_user(reg_user, reg_email, reg_pass):
                st.success("Registration successful! Please log in.")
            else:
                st.error("Username or email already exists.")

   