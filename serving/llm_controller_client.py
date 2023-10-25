import requests

def main():
    while True:
        string = input("Enter a command: ")
        requests.post('http://localhost:50001/command', json={'command': string})
        if string == "exit":
            break

if __name__ == "__main__":
    main()