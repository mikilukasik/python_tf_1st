from utils import print_large

print_large('Model saved.', './models/c4RESd2_S_v1/1.648378009001414')
print()

# Test with single argument
print_large("Hello world")
print()

# Test with multiple arguments
print_large("This is a test", "of the print_large function")
print()

# Test with numbers
print_large(1234567890, "This is a number", 42)
print()

# Test with long strings
print_large("This is a very long string that should still align",
            "with the border of the box")
print()

print_large("This is indented by 10", indent=10)
print_large("This is indented by 20", indent=20)
print_large("This is indented by 30", indent=30)
print()

print_large("Hello, World!")
print_large("123456789", "Hello, World!")
print_large(123, 456, 789)
print_large(1, "Hello", 2, "World", 3)
print_large("This is a long string that should be centered",
            "This is another long string that should be centered")
print_large("This is a short string", "This is another short string")
print_large("This is a long string that should be centered", 123,
            "This is another long string that should be centered", 456)
print_large("This is a short string", 123,
            "This is another short string", 45677)

print_large(1, 11, 111, 1111)
print_large('1', '11', '111', '1111')
