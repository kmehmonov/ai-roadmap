with open('copied.txt', encoding='utf8') as f:
    content = f.read()

content = content.replace('\n', '')

with open('copied.txt', 'w', encoding='utf8') as f:
    f.write(content)