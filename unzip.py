import zipfile

if __name__ == "__main__":
    with zipfile.ZipFile("./jhu_crowd_v2.0.zip", 'r') as file:
        file.extractall('./content/data')