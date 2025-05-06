import json

def ask(question):
    while True:
        answer = input(question + " (yes/no): ").strip().lower()
        if answer in ('yes', 'y'):
            return True
        elif answer in ('no', 'n'):
            return False
        else:
            print("Please answer yes or no.")

def traverse(tree):
    # If this is a leaf node (just an animal string), guess
    if isinstance(tree, str):
        print(f"Is it a {tree}?")
        return

    # Ask the current question
    answer = ask(tree['question'])

    # Follow the yes/no branch
    next_node = tree['yes'] if answer else tree['no']
    traverse(next_node)

def main():
    with open('questions.json', 'r') as f:
        tree = json.load(f)
    print("Think of an animal, and I will try to guess it!\n")
    traverse(tree)

if __name__ == '__main__':
    main()
