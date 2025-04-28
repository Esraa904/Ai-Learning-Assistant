# Function to print the suggestion
def suggest(content_type, reason):
    print(f"Recommendation: {content_type}")
    print(f"Reason: {reason}\n")

# Function to recommend material based on score and attempts
def recommend_material(score, attempts):
    if score < 50:
        suggest("easier content", f"Score is low ({score}), student needs simpler material.")
    elif score >= 85:
        suggest("advanced content", f"Score is high ({score}), student can handle advanced material.")
    else:
        suggest("keep current content", f"Score is moderate ({score}), continue at current level.")

# Main part: get input and run recommendation
def main():
    try:
        score = float(input("Enter student's score (0-100): "))
        attempts = int(input("Enter number of attempts: "))
        
        if score < 0 or score > 100 or attempts < 0:
            print("Invalid input. Please enter valid numbers.")
        else:
            recommend_material(score, attempts)
    except ValueError:
        print("Invalid input. Please enter numbers only.")

# Run the program
if __name__ == "__main__":
    main()
