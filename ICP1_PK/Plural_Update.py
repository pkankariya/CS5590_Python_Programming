import inflect
engine = inflect.engine()
user_sentence = input("Enter a sentence:")
plural = print(engine.plural(user_sentence))