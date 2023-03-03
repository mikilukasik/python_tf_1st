def print_large(*args, indent=30):
    """
    Prints a boxed text message with stars as a border and padding inside.

    Args:
        *args: Strings or numbers to be printed inside the box.
        indent: Optional integer representing the number of spaces to indent the box.

    Returns:
        None.
    """
    # Determine the longest string
    longest = str(max(args, key=lambda x: len(str(x))))
    # Calculate the width and height of the box
    box_width = len(longest) + 14
    # Define the top and bottom borders
    top_bottom_border = ('*' * (box_width)) + \
        ('*' if len(longest) % 2 == 0 else '')
    left_padding = ' ' * (indent)
    # Print the top border
    print(f'{left_padding}{top_bottom_border}')
    # Print the sides with text and padding
    for arg in args:
        arg_str = str(arg)
        arg_padding = ' ' * ((box_width - len(arg_str)) // 2 - 1)
        print(
            f'{left_padding}*{arg_padding}{arg_str}{arg_padding}{" " if len(arg_str) % 2 == 0 else ""}*')
    # Print the bottom border
    print(f'{left_padding}{top_bottom_border}')
