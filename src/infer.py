import streamlit as st

from src.model import get_model_fn
from src.utils.utils import parse_args

a = parse_args()
a.untrained = True

model = get_model_fn(a=a)()
model.eval()


def compute_score(list1, list2):
    return model(cv=[{"skills": list1}], job=[{"skills": list2}])


# Define the Streamlit app
def main():
    st.title("Compute Score")

    # Add input widgets for two lists
    input_list1 = st.text_input("Enter values for list 1, separated by commas", value="Python")
    input_list2 = st.text_input("Enter values for list 2, separated by commas", value="C++")

    # Add a button to compute the score
    if st.button("Compute Score") and input_list1 and input_list2:
        # Convert input strings to lists
        list1 = input_list1.split(",")
        list2 = input_list2.split(",")
        print(list1)
        print(list2)
        # Call the compute_score function and display the result
        score = compute_score(list1, list2)
        st.write("Score:", score)


# Run the app
if __name__ == "__main__":
    print(compute_score(["Python", "C++"], ["C++", "Java"]))
    main()
