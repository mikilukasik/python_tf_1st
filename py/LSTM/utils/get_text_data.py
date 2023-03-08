import os
import ebooklib
from ebooklib import epub
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def get_text_data(input_folder):
    file_extensions = ['.epub', '.pdf',  '.prc']

    # Function to recursively find all files with specified extensions in a folder
    def find_files(folder, extensions):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[-1] in extensions:
                    yield os.path.join(root, file)

    # Load the text from all files in the input folder with specified extensions
    book_text = ''
    for path in find_files(input_folder, file_extensions):
        print(path)
        book = None
        if path.endswith('.epub'):
            book = epub.read_epub(path)
        elif path.endswith('.pdf'):
            pdf_reader = PdfReader(path)
            for page in pdf_reader.pages:
                book_text += page.extract_text()
        if book is not None:
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                book_text += remove_html_tags(
                    item.get_body_content().decode("utf-8"))

    # print(book_text)
    return book_text


# get_text_data('../data_books')
