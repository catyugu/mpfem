import os

if __name__ == "__main__":
    os.system("ctest --test-dir build --output-on-failure")
    os.system("python -m pytest")
