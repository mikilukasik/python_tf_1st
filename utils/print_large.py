def print_large(*args, indent=30):
    """
    Prints a boxed text message with stars as a border and padding inside.

    Args:
        *args: Strings to be printed inside the box.
        indent: Optional integer representing the number of spaces to indent the box.

    Returns:
        None.
    """
    # Determine the longest string
    longest = max(args, key=len)
    # Calculate the width and height of the box
    box_width = len(longest) + 14
    # Define the top and bottom borders
    top_bottom_border = '*' * (box_width)
    left_padding = ' ' * (indent)
    # Print the top border
    print(f'{left_padding}{top_bottom_border}')
    # Print the sides with text and padding
    for text in args:
        text_padding = ' ' * ((box_width - len(text))//2 - 1)
        print(f'{left_padding}*{text_padding}{text}{text_padding}*')
    # Print the bottom border
    print(f'{left_padding}{top_bottom_border}')
