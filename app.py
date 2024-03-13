from transformers import TapasForQuestionAnswering, TapasTokenizer
import pandas as pd
import streamlit as st

# Set up TAPAS model and tokenizer
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

# Load the conversation table
conversation_table = pd.read_csv("Conversation.csv")

# Streamlit app
def main():
    st.title("TAPAS Model QA App")
    st.write("Ask a question and get an answer!")

    # Get user input for the question
    question = st.text_input("Ask a question (type 'exit' to quit): ")

    # Check if the user wants to exit
    if question.lower() == 'exit':
        st.stop()

    # Find the answer in the conversation table
    answer = find_answer_in_table(question, conversation_table)

    st.write(f"Question: {question}\nAnswer: {answer}\n")

def find_answer_in_table(question, table):
    # Filter the table to find the row matching the question
    matching_row = table[table['question'].str.lower() == question.lower()]

    # If a matching row is found, retrieve the corresponding answer
    if not matching_row.empty:
        answer = matching_row['answer'].iloc[0]
        return answer
    else:
        return "I'm sorry, I don't have an answer for that question."

if __name__ == "__main__":
    main()
