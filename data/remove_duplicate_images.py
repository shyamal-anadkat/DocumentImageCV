from difPy import dif

print('Deleting duplicate gujarati images...')
dif("raw/gujarati", delete=True, similarity="high")

print('Deleting duplicate russian images...')
dif("raw/russian", delete=True, similarity="high")

print('Deleting duplicate telugu images...')
dif("raw/telugu", delete=True, similarity="high")

print('Deleting duplicate english images...')
dif("raw/english", delete=True, similarity="high")