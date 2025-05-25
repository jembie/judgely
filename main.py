from court import Judge
from utils import launch


def main():
    launch()
    messages = [{"role": "user", "content": "How many r's are used in 'Erdbeere' and also 'Strawberry with cream'?"}]

    judge = Judge()
    reply = judge.chat(messages=messages)
    print(reply)


if __name__ == "__main__":
    main()
