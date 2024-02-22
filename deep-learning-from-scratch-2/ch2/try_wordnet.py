import nltk  # NLTK: Natural Language Toolkit
# nltk.download("wordnet")  # Download WordNet

from nltk.corpus import wordnet

def main():
    # Get a synset of a word: "car"
    print(wordnet.synsets("car"))

    # Print the meaning of the group: "car.n.01"
    car = wordnet.synset("car.n.01")
    print(car.definition())

    # Get synonyms in the group
    print(car.lemma_names())

    # Get hypernyms of the "car.n.01"
    # "entity" -> "physical_entity" -> "object" -> ... -> "motor_vehicle" -> "car"
    print(car.hypernym_paths()[0])

    # Get the similarity of words:
    # "novel", "dog", "motorcycle" and "car"
    novel = wordnet.synset("novel.n.01")
    dog = wordnet.synset("dog.n.01")
    motorcycle = wordnet.synset("motorcycle.n.01")
    print(car.path_similarity(novel))  # 0.056
    print(car.path_similarity(dog))  # 0.076
    print(car.path_similarity(motorcycle))  # 0.333


if __name__ == "__main__":
    main()
