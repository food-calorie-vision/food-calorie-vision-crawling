text = input("Enter the text to convert: ")

unicode_text = text.encode('unicode_escape').decode('utf-8')
print(unicode_text)
