def mapper(google):
    return str(google) + '-a', str(google) + '-b'

files = ['amazon', 'google', 'apple']

list = map(mapper, files)
for item in list:
    print(item)

print(list)
