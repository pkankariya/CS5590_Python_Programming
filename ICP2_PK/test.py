import inflect
engine = inflect.engine()
your_string= "i love dog"
plural = engine.plural(your_string)
print(plural)