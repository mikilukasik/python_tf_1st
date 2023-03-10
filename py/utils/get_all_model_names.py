import os


def get_all_model_names(folder, root=None):
    if root == None:
        root = folder

    model_names = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        if os.path.isdir(file_path):
            model_names.extend(get_all_model_names(file_path, root))
        elif file == "model.json":
            model_names.append(folder.replace(root, "", 1).lstrip(os.path.sep))

    return model_names
