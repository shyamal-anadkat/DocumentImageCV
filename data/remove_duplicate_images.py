from difPy import dif

print('Deleting duplicate russian images...')
dif("raw/data_to_transform/russian", delete=True, similarity="high")

print('Deleting duplicate telugu images...')
dif("raw/data_to_transform/telugu", delete=True, similarity="high")

print('Deleting duplicate english images...')
dif("raw/data_to_transform/english", delete=True, similarity="high")