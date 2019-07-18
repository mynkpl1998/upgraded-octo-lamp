import os

def clearCache():
    os.system("find . -name \"*.pyc\" -exec rm -rf {} \;")
    
if __name__ == "__main__":
    clearCache()
