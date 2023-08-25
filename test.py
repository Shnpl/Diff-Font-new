# detect if there're any sub directories in a specific style directory, which may
# cause exception
import os
root_dir = 'datasets/CFG/font500_6763'
for style_dir in os.listdir(root_dir):
    style_dir = os.path.join(root_dir, style_dir)
    items = os.listdir(style_dir)
    if len(items) != 6763:
        print(style_dir)
    for item in items:
        item = os.path.join(style_dir, item)
        if os.path.isdir(item):
            print(item)