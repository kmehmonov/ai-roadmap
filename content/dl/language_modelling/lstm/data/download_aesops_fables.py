import urllib.request

urllib.request.urlretrieve(
    'https://www.gutenberg.org/cache/epub/19994/pg19994.txt',
    'aesops_fables.txt'
)


# text_lines = text.splitlines()
# file_name = None
# file_body = None

# for line in text_lines:
#     if line.isupper():
#         if file_name and file_body:
#             with open("data\generated\\"+file_name, 'wt', encoding='utf-8') as f:
#                 f.write(file_body)
#             print('File Saved:', file_name)
            
#         file_name = line.lower().replace(' ', '_')+'.txt'
#         file_body = line + '\n'
#     else:
#         file_body += line + '\n'
    