from transformers import pipeline
classifier = pipeline("zero-shot-classification")
example = "This meatball recipe is spicy, try it!"
candidate_labels = ["cooking", "gardening", "blog", "techincal"]

print(classifier(example, candidate_labels))
